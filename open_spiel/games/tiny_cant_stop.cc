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

#include "open_spiel/games/tiny_cant_stop.h"

#include <sys/types.h>

#include <utility>

#include "open_spiel/game_parameters.h"
#include "open_spiel/spiel.h"

namespace open_spiel {
namespace tiny_cant_stop {

namespace {
// Moves.
enum ActionType { kRoll = 0, kStop = 1 };

// Default parameters.
constexpr int kDefaultColumnLength = 9;
constexpr int kDefaultDiceOutcomes = 6;
constexpr int kDefaultNumDice = 3;
constexpr int kDefaultSelectDice = 2;
constexpr int kDefaultTurnColumns = 3;
constexpr int kDefaultHorizon = 10000;
constexpr int kDefaultPlayers = 2;
constexpr int kDefaultWinScore = 2;
inline constexpr const char* kDefaultObservationEncoding = "one_hot";

// Facts about the game
const GameType kGameType{
    /*short_name=*/"tiny_cant_stop",
    /*long_name=*/"Tiny Can't Stop",
    GameType::Dynamics::kSequential,
    GameType::ChanceMode::kExplicitStochastic,
    GameType::Information::kPerfectInformation,
    GameType::Utility::kZeroSum,
    GameType::RewardModel::kTerminal,
    /*max_num_players=*/10, // with the right parameters
    /*min_num_players=*/2,
    /*provides_information_state_string=*/false,
    /*provides_information_state_tensor=*/false,
    /*provides_observation_string=*/true,
    /*provides_observation_tensor=*/true,
    /*parameter_specification=*/
    {
        {"columnlength", GameParameter(kDefaultColumnLength)},
        {"numdice", GameParameter(kDefaultNumDice)},
        {"selectdice", GameParameter(kDefaultSelectDice)},
        {"turncolumns", GameParameter(kDefaultTurnColumns)},
        {"players", GameParameter(kDefaultPlayers)},
        {"horizon", GameParameter(kDefaultHorizon)},
        {"winscore", GameParameter(kDefaultWinScore)},
        {"diceoutcomes", GameParameter(kDefaultDiceOutcomes)},
        {"observationencoding", GameParameter(static_cast<std::string>(kDefaultObservationEncoding))},
    }};

static std::shared_ptr<const Game> Factory(const GameParameters& params) {
  return std::shared_ptr<const Game>(new TinyCantStopGame(params));
}

REGISTER_SPIEL_GAME(kGameType, Factory);
}  // namespace

ObservationEncoding ParseObservationEncoding(const std::string& st_str) {
  if (st_str == "value") {
    return ObservationEncoding::kValue;
  } else if (st_str == "one_hot") {
    return ObservationEncoding::kOneHot;
  } else {
    SpielFatalError("Unrecognized observationencoding parameter: " + st_str);
  }
}

std::string TinyCantStopState::ActionToString(Player player, Action move_id) const {
  if (player == kChancePlayerId) {
    std::vector<int> rolls;
    RollDice(rolls, move_id);
    return absl::StrCat("Roll ", absl::StrJoin(rolls, " "));
  } else if (move_id == kRoll) {
    return "roll";
  } else if (move_id == kStop) {
    return "stop";
  } else {
    // Player selected a dice to progress
    int action_id_start = 2;
    // dice are chosen one by one, until no legal options are left, or until
    // the maximum number of allowed dice is reached
    // each 'selection round' gets its own action identifiers
    // starting from the 2nd selection round, the action_ids are increased
    for (int select_round = 0; select_round < n_dice_ - dice_.size(); select_round++) {
      // dont take the same action ids as the legal actions from previous rounds
      action_id_start += n_dice_ - select_round;
    }
    int dice_idx = move_id - action_id_start;
    return absl::StrCat("selected dice ", dice_idx, " with value ", dice_[dice_idx]);
  }
}

bool TinyCantStopState::IsTerminal() const {
  if (total_moves_ >= horizon_) {
    return true;
  }

  for (auto p = Player{0}; p < num_players_; p++) {
    if (scores_[p] >= win_score_) {
      return true;
    }
  }
  return false;
}

std::vector<double> TinyCantStopState::Returns() const {
  if (!IsTerminal()) {
    return std::vector<double>(num_players_, 0.0);
  }

  // For (n>2)-player games, must keep it zero-sum.
  std::vector<double> returns(num_players_, -1.0 / (num_players_ - 1));

  for (auto player = Player{0}; player < num_players_; ++player) {
    if (scores_[player] >= win_score_) {
      returns[player] = 1.0;
      return returns;
    }
  }

  // Nobody has won? (e.g. over horizon length.) Then everyone gets 0.
  return std::vector<double>(num_players_, 0.0);
}

std::string TinyCantStopState::ObservationString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  return ToString();
}

std::vector<int> TinyCantStopGame::ObservationTensorShape() const {
  switch (observation_encoding_) {
    case ObservationEncoding::kValue:
      return {1 + num_players_, dice_outcomes_};

    case ObservationEncoding::kOneHot:
      return {1 + num_players_, dice_outcomes_, col_len_ + 1};

    default:
      SpielFatalError("Unknown observation_encoding_");
  }
}

void TinyCantStopState::ObservationTensor(Player player,
                                 absl::Span<float> values) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);

  // TODO
  // NOTE: The observation tensor is always returned from the perspective of the
  // current player. It does NOT depend on the argument 'player' of this function.
  switch (observation_encoding_) {
    case ObservationEncoding::kValue:
    {
      // Value encoding: turn total value followed by p1, p2, ...
      SpielFatalError("Not yet implemented");
      break;
    }

    case ObservationEncoding::kOneHot:
    {
      SpielFatalError("Not yet implemented");
      break;
    }

    default:
      SpielFatalError("Unknown observation_encoding_");
  }
}

TinyCantStopState::TinyCantStopState(std::shared_ptr<const Game> game, int col_len, int dice_outcomes, int n_dice, int n_select, int turn_cols,
                   int horizon, int win_score, ObservationEncoding observation_encoding)
    : State(game),
      col_len_(col_len),
      dice_outcomes_(dice_outcomes),
      n_dice_(n_dice),
      n_select_(n_select),
      turn_cols_(turn_cols),
      horizon_(horizon),
      win_score_(win_score),
      observation_encoding_(observation_encoding),
      total_moves_(0),
      cur_player_(0),
      turn_player_(0),
      turn_progress_({}),
      dice_({}) {
  scores_.resize(game->NumPlayers(), 0);
  columns_.resize(dice_outcomes_, std::vector<int>(game->NumPlayers(), 0));
  turn_progress_.reserve(turn_cols_);
  dice_.reserve(n_dice);
}

int TinyCantStopState::CurrentPlayer() const {
  return IsTerminal() ? kTerminalPlayerId : cur_player_;
}

void TinyCantStopState::DoApplyAction(Action move) {
  // For decision node: 0 means roll, 1 means stop.
  // For chance node: outcome of the dice (0 for 1, 1 for 2+).
  if (cur_player_ >= 0) {
    if (move == kRoll) {
      // Player roll -> chance node.
      cur_player_ = kChancePlayerId;
      total_moves_++;
    } else if (move == kStop) {
      // Player stops. Take turn progress and pass to next player.
      LockProgress();
      turn_player_ = NextPlayerRoundRobin(turn_player_, num_players_);
      cur_player_ = turn_player_;
      total_moves_++;
    } else {
      // Player selected a dice to progress
      int action_id_start = 2;
      // dice are chosen one by one, until no legal options are left, or until
      // the maximum number of allowed dice is reached
      // each 'selection round' gets its own action identifiers
      // starting from the 2nd selection round, the action_ids are increased
      for (int select_round = 0; select_round < n_dice_ - dice_.size(); select_round++) {
        // dont take the same action ids as the legal actions from previous rounds
        action_id_start += n_dice_ - select_round;
      }
      int dice_idx = move - action_id_start;
      UpdateProgress(dice_[dice_idx]);
      // TODO: should we let the player choose an action, even if he couldn't progress
      // anymore with the rest of the dice or if all actions he can choose from lead to
      // the same result (i.e. only dice of the same value are left)?
      if (dice_.size() - 1 == n_dice_ - n_select_) {
        dice_.clear();
        // if dice_ is empty, the player can only decide between roll or stop next turn
      }
      else {
        dice_.erase(dice_.begin() + dice_idx);
        // dice_.size() > n_dice_ - n_select_
        // the player will still select further dice
      }
      total_moves_++;
    }
  } else if (IsChanceNode()) {
    // Resolve chance node outcome. If can't move, reset turn total and change players;
    // else, let the turn player select dice to move.
    RollDice(dice_, move);
    bool jeopardy = IsJeopardy();
    if (jeopardy) {
      // Reset turn total and loses turn!
      dice_.clear();
      turn_progress_.clear();
      turn_player_ = NextPlayerRoundRobin(turn_player_, num_players_);
      cur_player_ = turn_player_;
    } else {
      cur_player_ = turn_player_;
    }
  } else {
    SpielFatalError(absl::StrCat("Move ", move, " is invalid."));
  }
}

void TinyCantStopState::UpdateProgress(int column) {
  if (IsScored(column)) {
    return;
  }
  // the column is not scored yet
  for (ColumnProgress& cp : turn_progress_) {
    if (cp.column == column) {
      if ((columns_[cp.column][turn_player_] + cp.progress) < col_len_) {
        cp.progress += 1;
      }
      return;
    }
  }
  // the column does not exist in turn_progress_
  if (CanOpenNewCol()) {
    turn_progress_.push_back({
      /*column*/ column,
      /*progress*/ 1
    });
  }
  // if progress can't be used, do nothing
}

void TinyCantStopState::LockProgress() {
  for (ColumnProgress& cp : turn_progress_) {
    columns_[cp.column][turn_player_] += cp.progress;
    if (columns_[cp.column][turn_player_] == col_len_) {
      // score the column
      scores_[turn_player_] += 1;
    }
  }
  turn_progress_.clear();
}

bool TinyCantStopState::IsJeopardy() const {
  bool jeopardy = true;
  for (int d : dice_) {
    if (CanProgress(d))
      return false;
  }
  return true;
}

bool TinyCantStopState::CanProgress(int column) const {
  if (IsScored(column)) {
    return false;
  }
  for (ColumnProgress cp : turn_progress_) {
    // one of the player's progress columns, where progress is still possible
    if (column == cp.column) {
      if ((columns_[cp.column][turn_player_] + cp.progress) < col_len_) {
        return true;
      }
      // the player has already progressed to the end of the column
      return false;
    }
  }
  if (CanOpenNewCol()) {
    return true;
  }
  return false;
}

bool TinyCantStopState::IsScored(int column) const {
  for (int player_pos : columns_[column]) {
    if (player_pos == col_len_) {
      return true;
    }
  }
  return false;
}

bool TinyCantStopState::CanOpenNewCol() const {
  if (turn_progress_.size() < turn_cols_)
    return true;
  return false;
}

std::vector<Action> TinyCantStopState::LegalActions() const {
  if (IsChanceNode()) {
    return LegalChanceOutcomes();
  } else if (IsTerminal()) {
    return {};
  } else if (dice_.empty()) {
    // throw was evaluated or the player on turn changed -> choose whether to roll dice
    return {kRoll, kStop};
  } else {
    int action_id_start = 2;
    // dice were thrown -> choose where to advance
    // dice are chosen one by one, until no legal options are left, or until
    // the maximum number of allowed dice is reached
    // each 'selection round' gets its own action identifiers
    std::vector<Action> actions;
    // starting from the 2nd selection round, we need to increase the action_ids
    for (int select_round = 0; select_round < n_dice_ - dice_.size(); select_round++) {
      // dont take the same action ids as the legal actions from previous rounds
      action_id_start += n_dice_ - select_round;
    }
    for (int i = 0; i < dice_.size(); i++) {
      // don't force the player to choose a dice where progress is possible after the
      // first selection round
      if (n_dice_ != dice_.size() || CanProgress(dice_[i]))
        actions.push_back(action_id_start + i);
    }
    // assert that there are legal actions.
    // if there are none, the dice should have been cleared after the execution
    // of the roll (jeopardy).
    SPIEL_CHECK_GE(actions.size(), 1);
    return actions;
  }
}

std::vector<std::pair<Action, double>> TinyCantStopState::ChanceOutcomes() const {
  SPIEL_CHECK_TRUE(IsChanceNode());
  std::vector<std::pair<Action, double>> outcomes;

  int num_outcomes = 1;
  for (int i = 0; i < n_dice_; i++) {
    num_outcomes *= dice_outcomes_;
  }
  outcomes.reserve(num_outcomes);
  for (int i = 0; i < num_outcomes; i++) {
    outcomes.push_back(std::make_pair(i, 1.0 / num_outcomes));
  }

  return outcomes;
}

void TinyCantStopState::RollDice(std::vector<int>& dice_vec, Action outcome) const {
  SPIEL_CHECK_EQ(dice_vec.size(), 0);
  // Unrank action
  for (int d = n_dice_ - 1; d >= 0; d--) {
    dice_vec.push_back(outcome % dice_outcomes_);
    outcome /= dice_outcomes_;
  }
  SPIEL_CHECK_EQ(outcome, 0);
}

std::string TinyCantStopState::ToString() const {
  // TODO: print columns for more than 2 players
  std::string col_str = "";
  if (num_players_ == 2) {
    // for 2 players, use 'o' and 'x' to mark players progress
    // '*', if both players are on the same field
    std::vector<std::string> col_array;
    Player opp_player = 1 - turn_player_;
    std::vector<std::string> player_symbols = {"o", "x"};

    col_array.push_back(absl::StrCat("Player 0 = ", player_symbols[0],
                                     ", Player 1 = ", player_symbols[1]));
  
    for (int i = 0; i < dice_outcomes_; i++) {
      // print the columns as rows
      std::string cur_col = absl::StrCat(i, ": ");
      for (int j = 0; j <= col_len_; j++) {
        if (columns_[i][turn_player_] <= j && columns_[i][turn_player_] + getProgress(i) >= j) {
          if (columns_[i][opp_player] == j) {
            absl::StrAppend(&cur_col, "*");
          }
          else {
            absl::StrAppend(&cur_col, player_symbols[turn_player_]);
          }
        }
        else if (columns_[i][opp_player] == j) {
          absl::StrAppend(&cur_col, player_symbols[opp_player]);
        }
        else {
          absl::StrAppend(&cur_col, "-");
        }
      }
      if (IsScored(i)) {
        absl::StrAppend(&cur_col, " (scored)");
      }
      col_array.push_back(cur_col);
    }

    col_str = absl::StrJoin(col_array, "\n") + "\n";
  }
  std::string turn_total = "";
  for (ColumnProgress cp : turn_progress_) {
    turn_total = absl::StrCat(turn_total, " (", cp.column, ", ", cp.progress, ")");
  }

  std::string board_str = absl::StrCat(col_str, "Current player: ", turn_player_,
                                       (cur_player_ == kChancePlayerId ? " (rolling)\n" : "\n"));
  absl::StrAppend(&board_str, "Turn progress:", turn_total, "\n");
  absl::StrAppend(&board_str, "Dice: ", absl::StrJoin(dice_, " "), "\n");
  absl::StrAppend(&board_str, "Scores: ", absl::StrJoin(scores_, " "), "\n");

  return board_str;
}

int TinyCantStopState::getProgress(int column) const {
  for (ColumnProgress cp : turn_progress_) {
    if (cp.column == column) {
      return cp.progress;
    }
  }
  return 0;
}

std::unique_ptr<State> TinyCantStopState::Clone() const {
  return std::unique_ptr<State>(new TinyCantStopState(*this));
}

TinyCantStopGame::TinyCantStopGame(const GameParameters& params)
    : Game(kGameType, params),
      col_len_(ParameterValue<int>("columnlength")),
      dice_outcomes_(ParameterValue<int>("diceoutcomes")),
      n_dice_(ParameterValue<int>("numdice")),
      n_select_(ParameterValue<int>("selectdice")),
      turn_cols_(ParameterValue<int>("turncolumns")),
      horizon_(ParameterValue<int>("horizon")),
      num_players_(ParameterValue<int>("players")),
      win_score_(ParameterValue<int>("winscore")),
      observation_encoding_(
          ParseObservationEncoding(ParameterValue<std::string>("observationencoding"))) {}

}  // namespace tiny_cant_stop
}  // namespace open_spiel
