"""CPU regression tests for the source package."""

import ast
import importlib
from pathlib import Path
import re
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import torch

from datasets.CAVE_Dataset import cave_dataset
from datasets.Harvard_Dataset import harvard_dataset
from tools.Utils import para_setting


ROOT = Path(__file__).resolve().parents[1]


class ReleaseTests(unittest.TestCase):
    def test_psf_to_otf(self):
        for scale in (4, 8, 16, 32):
            with self.subTest(scale=scale):
                otf, adjoint = para_setting('gaussian_blur', scale, [64, 64], 2.0)
                kernel = cv2.getGaussianKernel(scale, 2.0)
                padded = np.zeros((64, 64))
                padded[:scale, :scale] = kernel @ kernel.T
                padded = np.roll(padded, (-scale // 2, -scale // 2), axis=(0, 1))
                np.testing.assert_allclose(otf, np.fft.fft2(padded), atol=1e-12)
                np.testing.assert_allclose(adjoint, np.conj(otf), atol=1e-12)

    def test_degradation_shapes_and_sampling_phase(self):
        image = torch.rand(1, 3, 64, 64, generator=torch.Generator().manual_seed(1))
        for scale in (4, 8, 16, 32):
            with self.subTest(scale=scale):
                otf, _ = para_setting('gaussian_blur', scale, [64, 64], 2.0)
                kernel = torch.from_numpy(np.stack([otf.real, otf.imag], axis=-1)).float()
                frequency = torch.complex(kernel[..., 0], kernel[..., 1])
                blurred = torch.fft.ifft2(torch.fft.fft2(image) * frequency).real
                phase = scale // 2 - 1
                expected = blurred[..., phase::scale, phase::scale]
                cave = cave_dataset.H_z(None, image, scale, kernel)
                harvard = harvard_dataset.H_z(image, scale, kernel)
                torch.testing.assert_close(cave, expected)
                torch.testing.assert_close(harvard, expected)
                torch.testing.assert_close(cave_dataset.H_z(None, image[0], scale, kernel), cave[0])

    def test_harvard_training_sample(self):
        # Broadcasted sources exercise crop selection without allocating full scenes.
        hsi = np.broadcast_to(np.float32(0.5), (1040, 1392, 31, 67))
        msi = np.broadcast_to(np.float32(0.5), (1040, 1392, 3, 67))
        opt = SimpleNamespace(data_path='', sf=4, trainset_num=1, sizeI=16)
        sample = harvard_dataset(opt, hsi, msi)[0]
        self.assertEqual([tuple(x.shape) for x in sample],
                         [(31, 4, 4), (3, 16, 16), (31, 16, 16), (16, 16, 2)])
        torch.testing.assert_close(sample[0], torch.full_like(sample[0], 0.5))

    def test_public_model_is_same_class(self):
        public = importlib.import_module('model.gsno')
        self.assertIs(public.GSNO, public.GSFusion)

    def test_release_scope(self):
        model_files = {p.relative_to(ROOT / 'model').as_posix()
                       for p in (ROOT / 'model').rglob('*.py')}
        self.assertEqual(model_files, {'gsno.py'})
        dataset_files = {p.name for p in (ROOT / 'datasets').glob('*.py')}
        self.assertEqual(dataset_files, {'CAVE_Dataset.py', 'Harvard_Dataset.py'})

    def test_training_defaults(self):
        tree = ast.parse((ROOT / 'Train_Cave.py').read_text(encoding='utf-8'))
        defaults = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'add_argument':
                if node.args and isinstance(node.args[0], ast.Constant):
                    for kw in node.keywords:
                        if kw.arg == 'default' and isinstance(kw.value, ast.Constant):
                            defaults[node.args[0].value] = kw.value.value
        self.assertEqual(defaults['--model'], 'gsno')
        self.assertEqual(defaults['--dim'], 80)
        self.assertEqual(defaults['--ep_total'], 1000)
        self.assertEqual(defaults['--seed'], 1)
        self.assertEqual(defaults['--sf'], 4)

    def test_evaluation_defaults(self):
        evaluator = importlib.import_module('tools.evaluate_dynamic_model_multiscale')
        argv = ['eval', '--checkpoint', 'weights.pth', '--data-path', 'data',
                '--output', 'result.json', '--selected-4x-best-epoch', '555',
                '--selected-4x-best-psnr', '52.6838439']
        with patch.object(sys, 'argv', argv):
            args = evaluator.parse_args()
        self.assertEqual(args.module, 'model.gsno')
        self.assertEqual(args.dim, 80)
        self.assertEqual(args.scales, [4, 8, 16, 32])

    def test_registered_modules_exist(self):
        tree = ast.parse((ROOT / 'Train_Cave.py').read_text(encoding='utf-8'))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Dict):
                continue
            for key, value in zip(node.keys, node.values):
                if isinstance(key, ast.Constant) and key.value == 'module' and isinstance(value, ast.Constant):
                    with self.subTest(module=value.value):
                        self.assertTrue((ROOT / (value.value.replace('.', '/') + '.py')).is_file())

    def test_document_links(self):
        for name in ('README.md', 'third_party/README.md', 'scripts/README.md',
                     'model/README.md'):
            path = ROOT / name
            for link in re.findall(r'\]\(([^)]+)\)', path.read_text(encoding='utf-8')):
                if '://' not in link and not link.startswith('#'):
                    self.assertTrue((path.parent / link.split('#')[0]).exists(), (name, link))
        self.assertTrue((ROOT / 'assets/gsno_framework.png').is_file())

    def test_license_copies(self):
        license_text = (ROOT / 'third_party/LICENSE_GAUSSIAN_SPLATTING.md').read_bytes()
        for directory in ('extensions/adaptive3_rasterizer',):
            self.assertEqual((ROOT / directory / 'LICENSE.md').read_bytes(), license_text)

    def test_shell_line_endings(self):
        for path in (ROOT / 'scripts').glob('*.sh'):
            with self.subTest(script=path.name):
                self.assertNotIn(b'\r', path.read_bytes())

    def test_command_help(self):
        for script in ('Train_Cave.py', 'Train_Harvard.py',
                       'tools/evaluate_dynamic_model_multiscale.py'):
            with self.subTest(script=script):
                result = subprocess.run([sys.executable, script, '--help'], cwd=ROOT,
                                        capture_output=True, text=True, timeout=60)
                self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == '__main__':
    unittest.main()
