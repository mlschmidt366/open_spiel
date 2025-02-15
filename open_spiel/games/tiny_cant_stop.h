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

#ifndef OPEN_SPIEL_GAMES_TINY_CANT_STOP_H_
#define OPEN_SPIEL_GAMES_TINY_CANT_STOP_H_

#include <memory>
#include <string>
#include <vector>

#include "open_spiel/spiel.h"

// A simple jeopardy dice game that includes chance nodes.
//
// Parameters:
//     "diceoutcomes"  int    number of outcomes of the dice  (default = 6)
//     "horizon"       int    max number of moves before draw (default = 1000)
//     "players"       int    number of players               (default = 2)
//     "winscore"      int    number of points needed to win   (default = 15)
//"observationencoding"str    one-hot or value encoding        (default = "value")

namespace open_spiel {
namespace tiny_cant_stop {

// How the observation should be encoded
enum class ObservationEncoding {
  kValue,  // "value": express progress with an integer.
  kOneHot,   // "one_hot": one-hot encoding.
};

struct ColumnProgress {
  int column;
  int progress;
};

class TinyCantStopGame;

class TinyCantStopState : public State {
 public:
  TinyCantStopState(const TinyCantStopState&) = default;
  TinyCantStopState(std::shared_ptr<const Game> game, int col_len, int dice_outcomes, int n_dice, int n_select, int turn_cols, int horizon,
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

  std::vector<Action> LegalActions() const override;

 protected:
  void DoApplyAction(Action move_id) override;

 private:
  // Compute the dice values based on the chance outcome
  void RollDice(std::vector<int> &dice_vec, Action outcome) const;

  // Update the turn progress with the newly selected column.
  void UpdateProgress(int column);

  // Advance the current player according to his progress in this turn.
  void LockProgress();

  // Whether the current dice jeopardy the current player
  bool IsJeopardy() const;

  // Whether the column was scored by any player
  bool IsScored(int column) const;

  // Whether the current player can progress on a new column
  bool CanOpenNewCol() const;

  // Whether the current player can progress in the specified column
  bool CanProgress(int column) const;

  // Get the current progress on the specified column
  // Returns 0, if it was not (yet) selected for progress
  int getProgress(int column) const;

  // Initialize to bad/invalid values. Use open_spiel::NewInitialState()
  int col_len_ = -1;
  int dice_outcomes_ = -1;  // Number of different dice outcomes (eg, 6).
  int n_dice_ = -1;
  int n_select_ = -1;
  int turn_cols_ = -1;
  int horizon_ = -1;
  int nplayers_ = -1;
  int win_score_ = 0;
  ObservationEncoding observation_encoding_; // which observation encoding to use (value or one-hot)

  int total_moves_ = -1;    // Total num moves taken during the game.
  Player cur_player_ = -1;  // Player to play.
  Player turn_player_ = -1;    // Whose actual turn is it. At chance nodes, we need
                            // to remember whose is playing for next turn.
                            // (cur_player will be the chance player's id.)
  std::vector<int> scores_;  // Score for each player.
  std::vector<std::vector<int>> columns_;
  std::vector<ColumnProgress> turn_progress_;
  std::vector<int> dice_;
};

class TinyCantStopGame : public Game {
 public:
  explicit TinyCantStopGame(const GameParameters& params);

  int NumDistinctActions() const override { return 2 + (n_dice_ * (n_dice_+1) ) / 2; }
  std::unique_ptr<State> NewInitialState() const override {
    return std::unique_ptr<State>(
        new TinyCantStopState(shared_from_this(), col_len_, dice_outcomes_, n_dice_, n_select_, turn_cols_, horizon_, win_score_, observation_encoding_));
  }
  int MaxChanceOutcomes() const override { return IntPower(dice_outcomes_, n_dice_); }

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
  // Length of each column.
  int col_len_;

  // Number of different dice outcomes, i.e. 6.
  int dice_outcomes_;

  // How many dice are rolled.
  int n_dice_;

  // How many dice to select each roll.
  int n_select_;

  // Maximum number of columns to progress in each turn.
  int turn_cols_;

  // Maximum number of moves before draw.
  int horizon_;

  // Number of players in this game.
  int num_players_;

  // The amount needed to win.
  int win_score_;

  // Which observation encoding to use (value or one-hot).
  ObservationEncoding observation_encoding_;
};

}  // namespace tiny_cant_stop
}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAMES_TINY_CANT_STOP_H_
