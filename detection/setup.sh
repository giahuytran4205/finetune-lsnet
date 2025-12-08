#!/bin/bash

# Dừng script ngay lập tức nếu có lỗi
set -e

echo ">>> BẮT ĐẦU SETUP..."

# 1. KÍCH HOẠT CONDA TRONG SCRIPT (QUAN TRỌNG NHẤT)
# Dựa vào log của bạn, đường dẫn cài đặt là /opt/miniforge3
if [ -f "/opt/miniforge3/etc/profile.d/conda.sh" ]; then
    source "/opt/miniforge3/etc/profile.d/conda.sh"
else
    echo "Lỗi: Không tìm thấy file conda.sh để kích hoạt!"
    exit 1
fi

# 2. Tạo môi trường Python 3.8 (Nếu chưa có)
# Thêm 'numpy' vào đây luôn để Conda cài bản chuẩn, tránh lỗi build bằng pip
echo ">>> Đang tạo môi trường lsnet..."
conda create -y -n lsnet python==3.8

# 3. Kích hoạt môi trường
echo ">>> Đang kích hoạt lsnet..."
conda activate lsnet

# Kiểm tra xem đã switch đúng python chưa
echo ">>> Python hiện tại: $(which python)"
echo ">>> Version: $(python --version)"

# 4. Cài các thư viện còn lại
# Lưu ý: Numpy đã cài ở trên rồi, pip sẽ tự bỏ qua hoặc check lại thôi
echo ">>> Đang cài requirements..."
pip install -r ../requirements.txt
pip install mmcv-full==1.7.2
pip install mmdet==2.28.2

cat > /venv/lsnet/lib/python3.8/site-packages/mmcv/parallel/_functions.py << 'EOF'
# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Union

import torch
from torch import Tensor
from torch.nn.parallel._functions import _get_stream

from packaging import version

def scatter(input: Union[List, Tensor],
            devices: List,
            streams: Optional[List] = None) -> Union[List, Tensor]:
    """Scatters tensor across multiple GPUs."""
    if streams is None:
        streams = [None] * len(devices)

    if isinstance(input, list):
        chunk_size = (len(input) - 1) // len(devices) + 1
        outputs = [
            scatter(input[i], [devices[i // chunk_size]],
                    [streams[i // chunk_size]]) for i in range(len(input))
        ]
        return outputs
    elif isinstance(input, Tensor):
        output = input.contiguous()
        # TODO: copy to a pinned buffer first (if copying from CPU)
        stream = streams[0] if output.numel() > 0 else None
        if devices != [-1]:
            with torch.cuda.device(devices[0]), torch.cuda.stream(stream):
                output = output.cuda(devices[0], non_blocking=True)

        return output
    else:
        raise Exception(f'Unknown type {type(input)}.')


def synchronize_stream(output: Union[List, Tensor], devices: List,
                       streams: List) -> None:
    if isinstance(output, list):
        chunk_size = len(output) // len(devices)
        for i in range(len(devices)):
            for j in range(chunk_size):
                synchronize_stream(output[i * chunk_size + j], [devices[i]],
                                   [streams[i]])
    elif isinstance(output, Tensor):
        if output.numel() != 0:
            with torch.cuda.device(devices[0]):
                main_stream = torch.cuda.current_stream()
                main_stream.wait_stream(streams[0])
                output.record_stream(main_stream)
    else:
        raise Exception(f'Unknown type {type(output)}.')


def get_input_device(input: Union[List, Tensor]) -> int:
    if isinstance(input, list):
        for item in input:
            input_device = get_input_device(item)
            if input_device != -1:
                return input_device
        return -1
    elif isinstance(input, Tensor):
        return input.get_device() if input.is_cuda else -1
    else:
        raise Exception(f'Unknown type {type(input)}.')


class Scatter:

    @staticmethod
    def forward(target_gpus: List[int], input: Union[List, Tensor]) -> tuple:
        input_device = get_input_device(input)
        streams = None
        if input_device == -1 and target_gpus != [-1]:
            # Perform CPU to GPU copies in a background stream
            if version.parse(torch.__version__) >= version.parse('2.1.0'):
                streams = [_get_stream(torch.device("cuda", device)) for device in target_gpus]
            else:
                streams = [_get_stream(device) for device in target_gpus]

        outputs = scatter(input, target_gpus, streams)
        # Synchronize with the copy stream
        if streams is not None:
            synchronize_stream(outputs, target_gpus, streams)

        return tuple(outputs) if isinstance(outputs, list) else (outputs, )
EOF

cat > /venv/lsnet/lib/python3.8/site-packages/mmcv/parallel/distributed.py << 'EOF'
# Copyright (c) OpenMMLab. All rights reserved.
from typing import Any, List, Tuple

import torch
from torch.nn.parallel.distributed import (DistributedDataParallel,
                                           _find_tensors)

from mmcv import print_log
from mmcv.utils import TORCH_VERSION, digit_version
from .scatter_gather import ScatterInputs, scatter_kwargs


class MMDistributedDataParallel(DistributedDataParallel):
    """The DDP module that supports DataContainer.

    MMDDP has two main differences with PyTorch DDP:

    - It supports a custom type :class:`DataContainer` which allows more
      flexible control of input data.
    - It implement two APIs ``train_step()`` and ``val_step()``.
    """

    def to_kwargs(self, inputs: ScatterInputs, kwargs: ScatterInputs,
                  device_id: int) -> Tuple[tuple, tuple]:
        # Use `self.to_kwargs` instead of `self.scatter` in pytorch1.8
        # to move all tensors to device_id
        return scatter_kwargs(inputs, kwargs, [device_id], dim=self.dim)

    def scatter(self, inputs: ScatterInputs, kwargs: ScatterInputs,
                device_ids: List[int]) -> Tuple[tuple, tuple]:
        return scatter_kwargs(inputs, kwargs, device_ids, dim=self.dim)

    def train_step(self, *inputs, **kwargs):
        """train_step() API for module wrapped by DistributedDataParallel.

        This method is basically the same as
        ``DistributedDataParallel.forward()``, while replacing
        ``self.module.forward()`` with ``self.module.train_step()``.
        It is compatible with PyTorch 1.1 - 1.5.
        """

        # In PyTorch >= 1.7, ``reducer._rebuild_buckets()`` is moved from the
        # end of backward to the beginning of forward.
        if ('parrots' not in TORCH_VERSION
                and digit_version(TORCH_VERSION) >= digit_version('1.7')
                and self.reducer._rebuild_buckets()):
            print_log(
                'Reducer buckets have been rebuilt in this iteration.',
                logger='mmcv')

        if ('parrots' not in TORCH_VERSION
                and digit_version(TORCH_VERSION) >= digit_version('1.11.0a0')):
            if self._check_sync_bufs_pre_fwd():
                self._sync_buffers()
        else:
            if (getattr(self, 'require_forward_param_sync', False)
                    and self.require_forward_param_sync):
                self._sync_params()

        if self.device_ids:
            inputs, kwargs = self.scatter(inputs, kwargs, self.device_ids)
            if len(self.device_ids) == 1:
                output = self.module.train_step(*inputs[0], **kwargs[0])
            else:
                outputs = self.parallel_apply(
                    self._module_copies[:len(inputs)], inputs, kwargs)
                output = self.gather(outputs, self.output_device)
        else:
            output = self.module.train_step(*inputs, **kwargs)

        if ('parrots' not in TORCH_VERSION
                and digit_version(TORCH_VERSION) >= digit_version('1.11.0a0')):
            if self._check_sync_bufs_post_fwd():
                self._sync_buffers()

        if (torch.is_grad_enabled()
                and getattr(self, 'require_backward_grad_sync', False)
                and self.require_backward_grad_sync):
            if self.find_unused_parameters:
                self.reducer.prepare_for_backward(list(_find_tensors(output)))
            else:
                self.reducer.prepare_for_backward([])
        else:
            if ('parrots' not in TORCH_VERSION
                    and digit_version(TORCH_VERSION) > digit_version('1.2')):
                self.require_forward_param_sync = False
        return output

    def val_step(self, *inputs, **kwargs):
        """val_step() API for module wrapped by DistributedDataParallel.

        This method is basically the same as
        ``DistributedDataParallel.forward()``, while replacing
        ``self.module.forward()`` with ``self.module.val_step()``.
        It is compatible with PyTorch 1.1 - 1.5.
        """
        # In PyTorch >= 1.7, ``reducer._rebuild_buckets()`` is moved from the
        # end of backward to the beginning of forward.
        if ('parrots' not in TORCH_VERSION
                and digit_version(TORCH_VERSION) >= digit_version('1.7')
                and self.reducer._rebuild_buckets()):
            print_log(
                'Reducer buckets have been rebuilt in this iteration.',
                logger='mmcv')

        if ('parrots' not in TORCH_VERSION
                and digit_version(TORCH_VERSION) >= digit_version('1.11.0a0')):
            if self._check_sync_bufs_pre_fwd():
                self._sync_buffers()
        else:
            if (getattr(self, 'require_forward_param_sync', False)
                    and self.require_forward_param_sync):
                self._sync_params()

        if self.device_ids:
            inputs, kwargs = self.scatter(inputs, kwargs, self.device_ids)
            if len(self.device_ids) == 1:
                output = self.module.val_step(*inputs[0], **kwargs[0])
            else:
                outputs = self.parallel_apply(
                    self._module_copies[:len(inputs)], inputs, kwargs)
                output = self.gather(outputs, self.output_device)
        else:
            output = self.module.val_step(*inputs, **kwargs)

        if ('parrots' not in TORCH_VERSION
                and digit_version(TORCH_VERSION) >= digit_version('1.11.0a0')):
            if self._check_sync_bufs_post_fwd():
                self._sync_buffers()

        if (torch.is_grad_enabled()
                and getattr(self, 'require_backward_grad_sync', False)
                and self.require_backward_grad_sync):
            if self.find_unused_parameters:
                self.reducer.prepare_for_backward(list(_find_tensors(output)))
            else:
                self.reducer.prepare_for_backward([])
        else:
            if ('parrots' not in TORCH_VERSION
                    and digit_version(TORCH_VERSION) > digit_version('1.2')):
                self.require_forward_param_sync = False
        return output

    def _run_ddp_forward(self, *inputs, **kwargs) -> Any:
        """Processes inputs and runs ``self.module.forward``.

        Pytorch 1.12.0 performs ``self.module.forward`` in ``_run_ddp_forward``
        and deprecates using ``DistributedDataParallel.to_kwargs`` to
        process inputs, which leads to inputs cannot be processed by
        :meth:`MMDistributedDataParallel.to_kwargs` anymore. Therefore,
        ``MMDistributedDataParallel`` overrides this method to call
        :meth:`to_kwargs` explicitly.

        See more information in `<https://github.com/open-mmlab/mmsegmentation/issues/1742>`_.  # noqa: E501

        Returns:
            Any: Forward result of :attr:`module`.
        """
        module_to_run = self.module

        if self.device_ids:
            inputs, kwargs = self.to_kwargs(  # type: ignore
                inputs, kwargs, self.device_ids[0])
            return module_to_run(*inputs[0], **kwargs[0])  # type: ignore
        else:
            return module_to_run(*inputs, **kwargs)
EOF

git clone https://huggingface.co/jameslahm/lsnet ./pretrain

git clone https://huggingface.co/giahuy4205/lsnet-finetuned-models ./pretrain

# 5. Tải data
echo ">>> Đang tải dataset..."
mkdir dataset
curl -L -o dataset/uecfoodpixcomplete-coco-format.zip \
    https://www.kaggle.com/api/v1/datasets/download/giahuytran1/uecfoodpixcomplete-coco-format

# 6. Giải nén
echo ">>> Đang giải nén..."
unzip -o -q dataset/uecfoodpixcomplete-coco-format.zip -d dataset/

echo ">>> HOÀN TẤT!"