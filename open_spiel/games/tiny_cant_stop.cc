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
constexpr int kDefaultDiceOutcomes = 6;
constexpr int kDefaultHorizon = 1000;
constexpr int kDefaultPlayers = 2;
constexpr int kDefaultWinScore = 15;
inline constexpr const char* kDefaultObservationEncoding = "value";

// Facts about the game
const GameType kGameType{
    /*short_name=*/"tiny_cant_stop",
    /*long_name=*/"Tiny Can't Stop",
    GameType::Dynamics::kSequential,
    GameType::ChanceMode::kExplicitStochastic,
    GameType::Information::kPerfectInformation,
    GameType::Utility::kZeroSum,
    GameType::RewardModel::kTerminal,
    /*max_num_players=*/10,
    /*min_num_players=*/2,
    /*provides_information_state_string=*/false,
    /*provides_information_state_tensor=*/false,
    /*provides_observation_string=*/true,
    /*provides_observation_tensor=*/true,
    /*parameter_specification=*/
    {
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
    return absl::StrCat("Roll ", move_id);
  } else if (move_id == kRoll) {
    return "roll";
  } else {
    return "stop";
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
      return {1 + num_players_};

    case ObservationEncoding::kOneHot:
      return {1 + num_players_, win_score_ + 1};

    default:
      SpielFatalError("Unknown observation_encoding_");
  }
}

void TinyCantStopState::ObservationTensor(Player player,
                                 absl::Span<float> values) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);

  switch (observation_encoding_) {
    case ObservationEncoding::kValue:
    {
      // Value encoding: turn total value followed by p1, p2, ...
      SPIEL_CHECK_EQ(values.size(), 1 + num_players_);
      values[0] = turn_total_;
      for (auto p = Player{0}; p < num_players_; p++) {
        values[1 + p] = scores_[p];
      }
      break;
    }

    case ObservationEncoding::kOneHot:
    {
      // One extra bin for when value is >= max.
      // So for win_score_ 15 -> 0, 1, ..., 99, >= 100.
      int num_bins = win_score_ + 1;

      // One-hot encoding: turn total (#bin) followed by p1, p2, ...
      SPIEL_CHECK_EQ(values.size(), num_bins + num_players_ * num_bins);
      std::fill(values.begin(), values.end(), 0.);
      int pos = 0;

      // One-hot encoding:
      //  - turn total (#bins)
      //  - player 0 (#bins)
      //  - player 1 (#bins)
      //      .
      //      .
      //      .

      int bin = turn_total_;
      if (bin >= num_bins) {
        // When the value is too large, use last bin.
        values[pos + (num_bins - 1)] = 1;
      } else {
        values[pos + bin] = 1;
      }

      pos += num_bins;

      // Find the right bin for each player.
      for (auto p = Player{0}; p < num_players_; p++) {
        bin = scores_[p];
        if (bin >= num_bins) {
          // When the value is too large, use last bin.
          values[pos + (num_bins - 1)] = 1;
        } else {
          values[pos + bin] = 1;
        }

        pos += num_bins;
      }
      break;
    }

    default:
      SpielFatalError("Unknown observation_encoding_");
  }
}

TinyCantStopState::TinyCantStopState(std::shared_ptr<const Game> game, int dice_outcomes,
                   int horizon, int win_score, ObservationEncoding observation_encoding)
    : State(game),
      dice_outcomes_(dice_outcomes),
      horizon_(horizon),
      win_score_(win_score),
      observation_encoding_(observation_encoding) {
  total_moves_ = 0;
  cur_player_ = 0;
  turn_player_ = 0;
  scores_.resize(game->NumPlayers(), 0);
  turn_total_ = 0;
}

int TinyCantStopState::CurrentPlayer() const {
  return IsTerminal() ? kTerminalPlayerId : cur_player_;
}

void TinyCantStopState::DoApplyAction(Action move) {
  // For decision node: 0 means roll, 1 means stop.
  // For chance node: outcome of the dice (0 for 1, 1 for 2+).
  if (cur_player_ >= 0 && move == kRoll) {
    SPIEL_CHECK_LT(scores_[cur_player_] + turn_total_, win_score_);
    // Player roll -> chance node.
    cur_player_ = kChancePlayerId;
    total_moves_++;
  } else if (cur_player_ >= 0 && move == kStop) {
    // Player stops. Take turn total and pass to next player.
    scores_[turn_player_] += turn_total_;
    turn_total_ = 0;
    turn_player_ = NextPlayerRoundRobin(turn_player_, num_players_);
    cur_player_ = turn_player_;
    total_moves_++;
  } else if (IsChanceNode()) {
    // Resolve chance node outcome. If 0, reset turn total and change players;
    // else, add to total and keep going.
    if (move == 0) {
      // Reset turn total and loses turn!
      turn_total_ = 0;
      turn_player_ = NextPlayerRoundRobin(turn_player_, num_players_);
      cur_player_ = turn_player_;
    } else {
      // Add to the turn total.
      turn_total_ += 1;
      cur_player_ = turn_player_;
    }
  } else {
    SpielFatalError(absl::StrCat("Move ", move, " is invalid."));
  }
}

std::vector<Action> TinyCantStopState::LegalActions() const {
  if (IsChanceNode()) {
    return LegalChanceOutcomes();
  } else if (IsTerminal()) {
    return {};
  } else {
    if (scores_[cur_player_] + turn_total_ >= win_score_) {
      return {kStop};
    } else {
      return {kRoll, kStop};
    }
  }
}

std::vector<std::pair<Action, double>> TinyCantStopState::ChanceOutcomes() const {
  SPIEL_CHECK_TRUE(IsChanceNode());
  std::vector<std::pair<Action, double>> outcomes;

  // Chance outcomes are labelled 0 or 1, corresponding to rolling 1 or 2+.
  outcomes.reserve(2);
  outcomes.push_back(std::make_pair(0, 1.0 / dice_outcomes_));
  outcomes.push_back(std::make_pair(1, 1.0 - 1.0 / dice_outcomes_));

  return outcomes;
}

std::string TinyCantStopState::ToString() const {
  return absl::StrCat("Scores: ", absl::StrJoin(scores_, " "),
                      ", Turn total: ", turn_total_,
                      "\nCurrent player: ", turn_player_,
                      (cur_player_ == kChancePlayerId ? " (rolling)\n" : "\n"));
}

std::unique_ptr<State> TinyCantStopState::Clone() const {
  return std::unique_ptr<State>(new TinyCantStopState(*this));
}

TinyCantStopGame::TinyCantStopGame(const GameParameters& params)
    : Game(kGameType, params),
      dice_outcomes_(ParameterValue<int>("diceoutcomes")),
      horizon_(ParameterValue<int>("horizon")),
      num_players_(ParameterValue<int>("players")),
      win_score_(ParameterValue<int>("winscore")),
      observation_encoding_(
          ParseObservationEncoding(ParameterValue<std::string>("observationencoding"))) {}

}  // namespace tiny_cant_stop
}  // namespace open_spiel
