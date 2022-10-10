#pragma once

#include <string>
#include <vector>

#include "../../youtokentome/cpp/bpe.h"
#include "../../youtokentome/cpp/third_party/flat_hash_map.h"

namespace vkcom {

flat_hash_map<uint32_t, uint32_t>
compute_alphabet_helper(const flat_hash_map<uint32_t, uint64_t> &char_cnt,
                        uint64_t data_len,
                        flat_hash_set<uint32_t> &removed_chars,
                        const BpeTrainConfig &bpe_config);

Status learn_bpe_from_string(std::string &text_utf8,
                             int n_tokens,
                             const std::string &output_file,
                             BpeTrainConfig bpe_config,
                             BpeState *bpe_state);

} // namespace vkcom
