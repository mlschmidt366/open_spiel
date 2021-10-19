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

#ifndef OPEN_SPIEL_GAMES_PIGLET_H_
#define OPEN_SPIEL_GAMES_PIGLET_H_

#include <memory>
#include <string>
#include <vector>

#include "open_spiel/spiel.h"

// A simple jeopardy dice game that includes chance nodes. Variation of Pig (see pig.h).
//
// Parameters:
//     "jeopardyprob"  int    probability of 'tails'          (default = 0.5)
//     "horizon"       int    max number of moves before draw (default = 400)
//     "players"       int    number of players               (default = 2)
//     "winscore"      int    number of points needed to win   (default = 10)
//"observationencoding"str    one-hot or value encoding        (default = "one_hot")

namespace open_spiel {
namespace piglet {

// How the observation should be encoded
enum class ObservationEncoding {
  kValue,  // "value": express progress with an integer.
  kOneHot,   // "one_hot": one-hot encoding.
};

class PigletGame;

class PigletState : public State {
 public:
  PigletState(const PigletState&) = default;
  PigletState(std::shared_ptr<const Game> game, double jeopardy_prob, int horizon,
           int win_score, ObservationEncoding observation_encoding);

  Player CurrentPlayer() const override;
  std::string ActionToString(Player player, Action move_id) const override;
  std::vector<std::pair<Action, double>> ChanceOutcomes() const override;
  std::string ToString() const override;
  bool IsTerminal() const override;
  std::vector<double> Returns() const override;
  std::string ObservationString(Player player) const override;
  void ObservationTensor(Player player,
                         absl::Span<float> values) const override;

  std::unique_ptr<State> Clone() const override;

  double jeopardy_prob() const { return jeopardy_prob_; }
  std::vector<Action> LegalActions() const override;

 protected:
  void DoApplyAction(Action move_id) override;

 private:
  // Initialize to bad/invalid values. Use open_spiel::NewInitialState()
  double jeopardy_prob_ = -1;  // Probability of throwing 'tails'.
  int horizon_ = -1;
  int nplayers_ = -1;
  int win_score_ = 0;
  ObservationEncoding observation_encoding_; // which observation encoding to use (value or one-hot)

  int total_moves_ = -1;    // Total num moves taken during the game.
  Player cur_player_ = -1;  // Player to play.
  int turn_player_ = -1;    // Whose actual turn is it. At chance nodes, we need
                            // to remember whose is playing for next turn.
                            // (cur_player will be the chance player's id.)
  std::vector<int> scores_;  // Score for each player.
  int turn_total_ = -1;
};

class PigletGame : public Game {
 public:
  explicit PigletGame(const GameParameters& params);

  int NumDistinctActions() const override { return 2; }
  std::unique_ptr<State> NewInitialState() const override {
    return std::unique_ptr<State>(
        new PigletState(shared_from_this(), jeopardy_prob_, horizon_, win_score_, observation_encoding_));
  }
  int MaxChanceOutcomes() const override { return 2; }

  // There is arbitrarily chosen number to ensure the game is finite.
  int MaxGameLength() const override { return horizon_; }
  // TODO: verify whether this bound is tight and/or tighten it.
  int MaxChanceNodesInHistory() const override { return MaxGameLength(); }

  int NumPlayers() const override { return num_players_; }
  double MinUtility() const override { return -1; }
  double UtilitySum() const override { return 0; }
  double MaxUtility() const override { return +1; }
  std::vector<int> ObservationTensorShape() const override;

 private:
  // Probability of throwing 'tails'.
  double jeopardy_prob_;

  // Maximum number of moves before draw.
  int horizon_;

  // Number of players in this game.
  int num_players_;

  // The amount needed to win.
  int win_score_;

  // Which observation encoding to use (value or one-hot).
  ObservationEncoding observation_encoding_;
};

}  // namespace piglet
}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAMES_PIGLET_H_
