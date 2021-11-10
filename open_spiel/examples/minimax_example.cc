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

#include "open_spiel/algorithms/minimax.h"
#include "open_spiel/algorithms/value_iteration.h"

#include <memory>

#include "open_spiel/games/breakthrough.h"
#include "open_spiel/games/pig.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"

inline constexpr int kSearchDepth = 2;
inline constexpr int kSearchDepthPig = 10;
inline constexpr int kWinscorePig = 30;
inline constexpr int kDiceoutcomesPig = 2;

uint_fast32_t Seed() {
  uint_fast32_t seed = 0;
  return seed != 0 ? seed : absl::ToUnixMicros(absl::Now());
}

namespace open_spiel {
namespace {

// expands a value solution over deterministic nodes to chance nodes
// can also be used to simply wrap a 'solution' in a function
std::function<double(const State&)> _chance_value_function(const std::map<std::string, double>& solution) {
  return [&solution](const State& state)->double {
    auto chance_evaluator = [&solution](const State& state, auto& evaluator_ref)->double {
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

void PlayGame(std::mt19937& rng) {
  std::shared_ptr<const Game> game =
      LoadGame("pig", {{"winscore", GameParameter(kWinscorePig)},
                      {"diceoutcomes", GameParameter(3)}});
  std::unique_ptr<State> state = game->NewInitialState();

  auto solution = algorithms::ValueIteration(*game, -1, 0.01);
  auto chance_solution = _chance_value_function(solution);

  while (!state->IsTerminal()) {
    std::cout << std::endl << state->ToString() << std::endl;
    Player player = state->CurrentPlayer();

    if (state->IsChanceNode()) {
      // Chance node; sample one according to underlying distribution.
      ActionsAndProbs outcomes = state->ChanceOutcomes();
      Action action = open_spiel::SampleAction(outcomes, rng).first;
      std::cerr << "Sampled action: " << state->ActionToString(player, action)
                << std::endl;
      state->ApplyAction(action);
    } else {
      auto value_function = [player, &chance_solution](const State& state)->double {
          return (player == Player{0} ? chance_solution(state) :
                                        -chance_solution(state));
          };
      std::pair<double, Action> value_action = algorithms::ExpectiminimaxSearch(
          *game, state.get(), value_function,
          /*depth_limit=*/1, player);

      std::cout << std::endl << "Player " << player << " choosing action "
                << state->ActionToString(player, value_action.second)
                << " with value " << value_action.first
                << std::endl;

      state->ApplyAction(value_action.second);
    }
  }

  std::cout << "Terminal state: " << std::endl;
  std::cout << state->ToString() << std::endl;
}


int BlackPieceAdvantage(const State& state) {
  const auto& bstate = down_cast<const breakthrough::BreakthroughState&>(state);
  return bstate.pieces(breakthrough::kBlackPlayerId) -
         bstate.pieces(breakthrough::kWhitePlayerId);
}

void PlayBreakthrough() {
  std::shared_ptr<const Game> game =
      LoadGame("breakthrough", {{"rows", GameParameter(6)},
                                {"columns", GameParameter(6)}});
  std::unique_ptr<State> state = game->NewInitialState();
  while (!state->IsTerminal()) {
    std::cout << std::endl << state->ToString() << std::endl;

    Player player = state->CurrentPlayer();
    std::pair<double, Action> value_action = algorithms::AlphaBetaSearch(
        *game, state.get(), [player](const State& state) {
            return (player == breakthrough::kBlackPlayerId ?
                    BlackPieceAdvantage(state) :
                    -BlackPieceAdvantage(state));
            },
        kSearchDepth, player);

    std::cout << std::endl << "Player " << player << " choosing action "
              << state->ActionToString(player, value_action.second)
              << " with heuristic value (to black) " << value_action.first
              << std::endl;

    state->ApplyAction(value_action.second);
  }

  std::cout << "Terminal state: " << std::endl;
  std::cout << state->ToString() << std::endl;
}


int FirstPlayerAdvantage(const State& state) {
  const auto& pstate = down_cast<const pig::PigState&>(state);
  return pstate.score(0) - pstate.score(1);
}

void PlayPig(std::mt19937& rng) {
  std::shared_ptr<const Game> game =
      LoadGame("pig", {{"winscore", GameParameter(kWinscorePig)},
                      {"diceoutcomes", GameParameter(kDiceoutcomesPig)}});
  std::unique_ptr<State> state = game->NewInitialState();
  while (!state->IsTerminal()) {
    std::cout << std::endl << state->ToString() << std::endl;

    Player player = state->CurrentPlayer();
    if (state->IsChanceNode()) {
      // Chance node; sample one according to underlying distribution.
      ActionsAndProbs outcomes = state->ChanceOutcomes();
      Action action = open_spiel::SampleAction(outcomes, rng).first;
      std::cerr << "Sampled action: " << state->ActionToString(player, action)
                << std::endl;
      state->ApplyAction(action);
    } else {
      std::pair<double, Action> value_action = algorithms::ExpectiminimaxSearch(
          *game, state.get(), [player](const State& state) {
              return (player == Player{0} ?
                      FirstPlayerAdvantage(state) :
                      -FirstPlayerAdvantage(state));
              },
          kSearchDepthPig, player);

      std::cout << std::endl << "Player " << player << " choosing action "
                << state->ActionToString(player, value_action.second)
                << " with heuristic value " << value_action.first
                << std::endl;

      state->ApplyAction(value_action.second);
    }
  }

  std::cout << "Terminal state: " << std::endl;
  std::cout << state->ToString() << std::endl;
}

}  // namespace
}  // namespace open_spiel

int main(int argc, char **argv) {
  std::mt19937 rng(Seed());  // Random number generator.
  open_spiel::PlayBreakthrough();
  open_spiel::PlayPig(rng);
}
