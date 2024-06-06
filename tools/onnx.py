# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Export a trained model from the full checkpoint (with optimizer etc.) to
a final checkpoint, with only the model itself. The model is always stored as
half float to gain space, and because this has zero impact on the final loss.
When DiffQ was used for training, the model will actually be quantized and bitpacked."""
from argparse import ArgumentParser
from fractions import Fraction
import logging
from pathlib import Path
import sys
import torch

from demucs import train, pretrained
from demucs.states import serialize_model, save_with_checksum

from einops._torch_specific import allow_ops_in_compiled_graph  # requires einops>=0.6.1
allow_ops_in_compiled_graph()

logger = logging.getLogger(__name__)


def main():
    logging.basicConfig(level=logging.INFO, stream=sys.stderr)

    parser = ArgumentParser("tools.export", description="Export trained models from XP sigs.")
    parser.add_argument('signatures', nargs='*', help='XP signatures.')
    parser.add_argument('-o', '--out', type=Path, default=Path("release_models"),
                        help="Path where to store release models (default release_models)")
    parser.add_argument('-s', '--sign', action='store_true',
                        help='Add sha256 prefix checksum to the filename.')

    args = parser.parse_args()
    args.out.mkdir(exist_ok=True, parents=True)

    for sig in args.signatures:
        model = pretrained.get_model(sig)
        model = model.models[0]
        
        #print(model)
        
        dummy_input = (torch.randn(1, 2, 343980), torch.randn(1,4,2048,336))
        
        torch.onnx.export(model, dummy_input, sig + ".onnx", input_names=["mix", "stft"], output_names=["masks", "xt"])
        
        #stft_input = torch.randn((1,4,2048,336)) #, dtype=torch.cfloat)#, torch.randn(1,2,2048,336))
#        onnx_model = torch.onnx.dynamo_export(model, mix_input, stft_input)
#        onnx_model = torch.onnx.dynamo_export(model, mix_input)
#        onnx_model.save(sig + "2.onnx")


if __name__ == '__main__':
    main()
