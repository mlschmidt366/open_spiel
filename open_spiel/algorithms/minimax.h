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

#ifndef OPEN_SPIEL_ALGORITHMS_MINMAX_H_
#define OPEN_SPIEL_ALGORITHMS_MINMAX_H_

#include <memory>
#include <utility>

#include "open_spiel/spiel.h"
#include "open_spiel/spiel_bots.h"

namespace open_spiel {
namespace algorithms {

// Solves deterministic, 2-players, perfect-information 0-sum game.
//
// For small games only! Please use keyword arguments for optional arguments.
//
// Arguments:
//   game: The game to analyze, as returned by `LoadGame`.
//   state: The state to start from. If nullptr, starts from initial state.
//   value_function: An optional function mapping a Spiel `State` to a
//     numerical value to the maximizing player, to be used as the value for a
//     node when we reach `depth_limit` and the node is not terminal. Use
//     `nullptr` for no value function.
//   depth_limit: The maximum depth to search over. When this depth is
//     reached, an exception will be raised.
//   maximizing_player_id: The id of the MAX player. The other player is assumed
//     to be MIN. Passing in kInvalidPlayer will set this to the search root's
//     current player.

//   Returns:
//     A pair of the value of the game for the maximizing player when both
//     players play optimally, along with the action that achieves this value.

std::pair<double, Action> AlphaBetaSearch(
    const Game& game, const State* state,
    std::function<double(const State&)> value_function, int depth_limit,
    Player maximizing_player);


// Solves stochastic, 2-players, perfect-information 0-sum game.
//
// For small games only! Please use keyword arguments for optional arguments.
//
// Arguments:
//   game: The game to analyze, as returned by `LoadGame`.
//   state: The state to start from. If nullptr, starts from initial state.
//   value_function: An optional function mapping a Spiel `State` to a
//     numerical value to the maximizing player, to be used as the value for a
//     node when we reach `depth_limit` and the node is not terminal. Use
//     `nullptr` for no value function.
//   depth_limit: The maximum depth to search over (not counting chance nodes).
//     When this depth is reached, an exception will be raised.
//   maximizing_player_id: The id of the MAX player. The other player is assumed
//     to be MIN. Passing in kInvalidPlayer will set this to the search root's
//     current player.

//   Returns:
//     A pair of the value of the game for the maximizing player when both
//     players play optimally, along with the action that achieves this value.

std::pair<double, Action> ExpectiminimaxSearch(
    const Game& game, const State* state,
    std::function<double(const State&)> value_function, int depth_limit,
    Player maximizing_player);


// A SpielBot that uses one of the Minimax algorithms as its policy.
// AlphaBetaSearch for deterministic games and ExpectiminimaxSearch for
// explicit stochastic games.
class MinimaxBot : public Bot {
 public:
  MinimaxBot(
      const Game& game, std::function<double(const State&)> value_function,
      int depth_limit, Player maximizing_player, bool verbose);
  ~MinimaxBot() = default;

  void Restart() override {}
  void RestartAt(const State& state) override {}
  // Run Minimax for one step, choosing the action, and printing some information.
  Action Step(const State& state) override;

  // Implements StepWithPolicy. This is equivalent to calling Step, but wraps
  // the action as an ActionsAndProbs with 100% probability assigned to the
  // lone action.
  std::pair<ActionsAndProbs, Action> StepWithPolicy(
      const State& state) override;

  // Run Minimax on a given state, and return the value and best action.
  std::pair<double, Action> MinimaxSearch(const State& state);

 private:
  int depth_limit_;
  bool verbose_;
  Player maximizing_player_;
  std::function<double(const State&)> value_function_;
  bool deterministic_game_;
};

}  // namespace algorithms
}  // namespace open_spiel

#endif  // OPEN_SPIEL_ALGORITHMS_MINMAX_H_
