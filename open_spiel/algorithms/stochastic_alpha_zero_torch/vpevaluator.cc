// Copyright 2019 DeepMind Technologies Ltd. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "open_spiel/algorithms/stochastic_alpha_zero_torch/vpevaluator.h"

#include <cstdint>
#include <memory>

#include "open_spiel/abseil-cpp/absl/hash/hash.h"
#include "open_spiel/abseil-cpp/absl/time/time.h"
#include "open_spiel/utils/stats.h"

namespace open_spiel {
namespace algorithms {
namespace torch_az {

VPNetEvaluator::VPNetEvaluator(DeviceManager* device_manager, int batch_size,
                               int threads, int cache_size, int cache_shards)
    : device_manager_(*device_manager),
      batch_size_(batch_size),
      queue_(batch_size * threads * 4),
      batch_size_hist_(batch_size + 1) {
  cache_shards = std::max(1, cache_shards);
  cache_.reserve(cache_shards);
  for (int i = 0; i < cache_shards; ++i) {
    cache_.push_back(
        std::make_unique<LRUCache<uint64_t, VPNetModel::InferenceOutputs>>(
            cache_size / cache_shards));
  }
  if (batch_size_ <= 1) {
    threads = 0;
  }
  inference_threads_.reserve(threads);
  for (int i = 0; i < threads; ++i) {
    inference_threads_.emplace_back([this]() { this->Runner(); });
  }
}

VPNetEvaluator::~VPNetEvaluator() {
  stop_.Stop();
  queue_.BlockNewValues();
  queue_.Clear();
  for (auto& t : inference_threads_) {
    t.join();
  }
}

void VPNetEvaluator::ClearCache() {
  for (auto& c : cache_) {
    c->Clear();
  }
}

LRUCacheInfo VPNetEvaluator::CacheInfo() {
  LRUCacheInfo info;
  for (auto& c : cache_) {
    info += c->Info();
  }
  return info;
}

std::vector<double> VPNetEvaluator::Evaluate(const State& state) {
  if (state.IsChanceNode()) {
    /* TODO: include later
    std::unique_ptr<State> working_state = state.Clone();
    open_spiel::Action last_action = working_state->UndoLastAction();
    if (working_state->IsChanceNode()) {
      open_spiel::SpielFatalError("AlphaZero can only handle one chance node at a time.");
    }
    double p0value = Inference(working_state).action_values[last_action];
    return {p0value, -p0value};
    */
    // TEST with random rollout at chance nodes
    std::cerr << "AlphaZero shouldn't need to evaluate Chance Nodes currently. (Probably error?)" << std::endl;
    auto test_evaluator =
      std::make_shared<open_spiel::algorithms::RandomRolloutEvaluator>(
          /*rollout_count*/10, /*seed*/0);
    return test_evaluator->Evaluate(state);
  } else {
    // TODO(author5): currently assumes zero-sum.
    // Make sure that the observationTensor is returned from the perspective
    // of the current player in the implementation of each game!
    double cur_player_value = Inference(state).value;
    if (state.CurrentPlayer() == 1) {
      return {-cur_player_value, cur_player_value};
    }
    return {cur_player_value, -cur_player_value};
  }
}

open_spiel::ActionsAndProbs VPNetEvaluator::Prior(const State& state) {
  // return chance values for chance nodes, inferred policy otherwise
  if (state.IsChanceNode()) {
    return state.ChanceOutcomes();
  } else {
    // the state is from the perspective of the current player
    return Inference(state).policy;
  }
}

VPNetModel::InferenceOutputs VPNetEvaluator::Inference(const State& state) {
  VPNetModel::InferenceInputs inputs = {state.LegalActions(),
                                        state.ObservationTensor()};

  // DONE: test if ObservationTensor changes perspective
  // - It does not (at least for Tic Tac Toe)
  //std::cout << "ObservationTensor " << state.ObservationTensor() << std::endl;

  uint64_t key;
  int cache_shard;
  if (!cache_.empty()) {
    key = absl::Hash<VPNetModel::InferenceInputs>{}(inputs);
    cache_shard = key % cache_.size();
    absl::optional<const VPNetModel::InferenceOutputs> opt_outputs =
        cache_[cache_shard]->Get(key);
    if (opt_outputs) {
      return *opt_outputs;
    }
  }
  VPNetModel::InferenceOutputs outputs;
  if (batch_size_ <= 1) {
    outputs = device_manager_.Get(1)->Inference(std::vector{inputs})[0];
  } else {
    std::promise<VPNetModel::InferenceOutputs> prom;
    std::future<VPNetModel::InferenceOutputs> fut = prom.get_future();
    queue_.Push(QueueItem{inputs, &prom});
    outputs = fut.get();
  }
  if (!cache_.empty()) {
    cache_[cache_shard]->Set(key, outputs);
  }
  return outputs;
}

void VPNetEvaluator::Runner() {
  std::vector<VPNetModel::InferenceInputs> inputs;
  std::vector<std::promise<VPNetModel::InferenceOutputs>*> promises;
  inputs.reserve(batch_size_);
  promises.reserve(batch_size_);
  while (!stop_.StopRequested()) {
    {
      // Only one thread at a time should be listening to the queue to maximize
      // batch size and minimize latency.
      absl::MutexLock lock(&inference_queue_m_);
      absl::Time deadline = absl::InfiniteFuture();
      for (int i = 0; i < batch_size_; ++i) {
        absl::optional<QueueItem> item = queue_.Pop(deadline);
        if (!item) {  // Hit the deadline.
          break;
        }
        if (inputs.empty()) {
          deadline = absl::Now() + absl::Milliseconds(1);
        }
        inputs.push_back(item->inputs);
        promises.push_back(item->prom);
      }
    }

    if (inputs.empty()) {  // Almost certainly StopRequested.
      continue;
    }

    {
      absl::MutexLock lock(&stats_m_);
      batch_size_stats_.Add(inputs.size());
      batch_size_hist_.Add(inputs.size());
    }

    std::vector<VPNetModel::InferenceOutputs> outputs =
        device_manager_.Get(inputs.size())->Inference(inputs);
    for (int i = 0; i < promises.size(); ++i) {
      promises[i]->set_value(outputs[i]);
    }
    inputs.clear();
    promises.clear();
  }
}

void VPNetEvaluator::ResetBatchSizeStats() {
  absl::MutexLock lock(&stats_m_);
  batch_size_stats_.Reset();
  batch_size_hist_.Reset();
}

open_spiel::BasicStats VPNetEvaluator::BatchSizeStats() {
  absl::MutexLock lock(&stats_m_);
  return batch_size_stats_;
}

open_spiel::HistogramNumbered VPNetEvaluator::BatchSizeHistogram() {
  absl::MutexLock lock(&stats_m_);
  return batch_size_hist_;
}

}  // namespace torch_az
}  // namespace algorithms
}  // namespace open_spiel
