/* Copyright 2017 Battelle Energy Alliance, LLC

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
*/
#include "randomClass.h"
#include <boost/random/mersenne_twister.hpp>

class RandomClassImpl {
  /**
   * This class create the instance of a mersenne-twister random number generator (BOOST)
   */
public:
  boost::random::mt19937 _backend;
};

RandomClass::RandomClass() : _rng(new RandomClassImpl()), _range(_rng->_backend.max() - _rng->_backend.min()) {
    _counter=0;
    _seed=0;
}

void RandomClass::seed(unsigned long int seed) {

    _counter = 0;
    _seed = seed;
    _rng->_backend.seed(seed);
  }

double RandomClass::random() {
    _counter++;
    return (_rng->_backend()-_rng->_backend.min())/_range;
  }

int RandomClass::get_rng_state() {
    return _counter;
}

void RandomClass::forward_seed(unsigned int counts){
    _counter = counts;
    _rng->_backend.discard(counts);
}

int RandomClass::get_rng_seed(){
  return _seed;
}

RandomClass::~RandomClass(){
  delete _rng;
}
