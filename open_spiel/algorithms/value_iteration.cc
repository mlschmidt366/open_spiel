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

#include "open_spiel/algorithms/value_iteration.h"

#include <algorithm>

#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace algorithms {
namespace {

using std::map;
using std::vector;
using state_pointer = std::unique_ptr<State>;
using state_action = std::pair<std::string, Action>;
using state_prob = std::pair<std::string, double>;

// Adds transitions and transition probability from a given state
void AddTransition(map<state_action, vector<state_prob>>* transitions,
                   std::string key, const state_pointer& state,
                   bool include_chance_states) {
  if (state->IsChanceNode()) {
    SPIEL_CHECK_EQ(include_chance_states, true);
    for (const auto& actionprob : state->ChanceOutcomes()) {
      vector<state_prob> possibilities;
      // For a chance node, record the transition probabilities
      auto next_state = state->Clone();
      next_state->ApplyAction(actionprob.first);
      possibilities.emplace_back(next_state->ToString(),
                                  actionprob.second);
      (*transitions)[std::make_pair(key, actionprob.first)] = possibilities;
    }
  } else {
    for (auto action : state->LegalActions()) {
      auto next_state = state->Clone();
      next_state->ApplyAction(action);
      vector<state_prob> possibilities;
      if (next_state->IsChanceNode() && !include_chance_states) {
        // For a chance node, record the transition probabilities
        for (const auto& actionprob : next_state->ChanceOutcomes()) {
          auto realized_next_state = next_state->Clone();
          realized_next_state->ApplyAction(actionprob.first);
          possibilities.emplace_back(realized_next_state->ToString(),
                                    actionprob.second);
        }
      } else {
        // A non-chance node is equivalent to transition with probability 1
        possibilities.emplace_back(next_state->ToString(), 1.0);
      }
      (*transitions)[std::make_pair(key, action)] = possibilities;
    }
  }
}

// Initialize transition map and value map
void InitializeMaps(const map<std::string, state_pointer>& states,
                    map<std::string, double>* values,
                    map<state_action, vector<state_prob>>* transitions,
                    bool include_chance_states) {
  for (const auto& kv : states) {
    auto key = kv.first;
    if (kv.second->IsTerminal()) {
      // For both 1-player and 2-player zero sum games, suffices to look at
      // player 0's utility
      (*values)[key] = kv.second->PlayerReturn(Player{0});
    } else {
      (*values)[key] = 0;
      AddTransition(transitions, key, kv.second, include_chance_states);
    }
  }
}

}  // namespace

std::map<std::string, double> ValueIteration(const Game& game, int depth_limit,
                                             double threshold, bool include_chance_states) {
  using state_action = std::pair<std::string, Action>;
  using state_prob = std::pair<std::string, double>;

  // Currently only supports 1-player or 2-player zero sum games
  SPIEL_CHECK_TRUE(game.NumPlayers() == 1 || game.NumPlayers() == 2);
  if (game.NumPlayers() == 2) {
    SPIEL_CHECK_EQ(game.GetType().utility, GameType::Utility::kZeroSum);
  }

  // No support for simultaneous games (needs an LP solver). And so also must
  // be a perfect information game.
  SPIEL_CHECK_EQ(game.GetType().dynamics, GameType::Dynamics::kSequential);
  SPIEL_CHECK_EQ(game.GetType().information,
                 GameType::Information::kPerfectInformation);

  auto states = GetAllStates(game, depth_limit, /*include_terminals=*/true,
                             include_chance_states);
  std::map<std::string, double> values;
  std::map<state_action, std::vector<state_prob>> transitions;

  InitializeMaps(states, &values, &transitions, include_chance_states);

  double error;
  double min_utility = game.MinUtility();
  double max_utility = game.MaxUtility();
  do {
    error = 0;
    for (const auto& kv : states) {
      auto key = kv.first;

      if (kv.second->IsTerminal()) continue;

      auto player = kv.second->CurrentPlayer();

      // Initialize value to be 0 if chance node, or else
      // to be the minimum utility if current player
      // is the maximizing player (i.e. player 0), and to maximum utility
      // if current player is the minimizing player (i.e. player 1).
      double value = (player == kChancePlayerId) ? 0 :
                     (player == Player{0})       ? min_utility :
                                                   max_utility;
      for (auto action : kv.second->LegalActions()) {
        auto possibilities = transitions[std::make_pair(key, action)];
        double q_value = 0;
        for (const auto& outcome : possibilities) {
          q_value += outcome.second * values[outcome.first];
        }
        // Chance Player is averaging the value
        // Player 0 is maximizing the value (which is w.r.t. player 0)
        // Player 1 is minimizing the value
        if (player == kChancePlayerId)
          value += q_value;
        else if (player == Player{0})
          value = std::max(value, q_value);
        else
          value = std::min(value, q_value);
      }

      double* stored_value = &values[key];
      error = std::max(std::abs(*stored_value - value), error);
      *stored_value = value;
    }
  } while (error > threshold);

  return values;
}

std::function<double(const State&)> MakeChanceValueFunction(const std::map<std::string, double>& solution) {
  // capture 'solution' by copy, so that it is available even after the reference from the calling scope
  // is deallocated
  return [solution](const State& state)->double {
    auto chance_evaluator = [&solution](const State& state, const auto& evaluator_ref)->double {
      if (state.IsChanceNode() && solution.find(state.ToString()) == solution.end()) {
        double value = 0;
        for (const auto& actionprob : state.ChanceOutcomes()) {
          auto child_state = state.Clone();
          child_state->ApplyAction(actionprob.first);
          double child_value = evaluator_ref(*child_state, evaluator_ref);
          value += actionprob.second * child_value;
        }
        return value;
      }
      return solution.at(state.ToString());
    };
    return chance_evaluator(state, chance_evaluator);
  };
}

}  // namespace algorithms
}  // namespace open_spiel
