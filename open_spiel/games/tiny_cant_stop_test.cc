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

#include "open_spiel/spiel.h"
#include "open_spiel/tests/basic_tests.h"

namespace open_spiel {
namespace tiny_cant_stop {
namespace {

namespace testing = open_spiel::testing;

void BasicTinyCantStopTests() {
  testing::LoadGameTest("tiny_cant_stop");
  testing::ChanceOutcomesTest(*LoadGame("tiny_cant_stop"));
  testing::RandomSimTest(*LoadGame("tiny_cant_stop"), 100);
  for (Player players = 3; players <= 5; players++) {
    testing::RandomSimTest(
        *LoadGame("tiny_cant_stop", {{"players", GameParameter(players)}}), 100);
  }
  std::string enc = "one_hot";
  testing::RandomSimTest(
      *LoadGame("tiny_cant_stop", {{"observationencoding", GameParameter(enc)}}), 100);
}

}  // namespace
}  // namespace tiny_cant_stop
}  // namespace open_spiel

int main(int argc, char** argv) { open_spiel::tiny_cant_stop::BasicTinyCantStopTests(); }
