import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

from .dataset_versions import default_release, validate_dataset_version


class DatasetVersionTests(unittest.TestCase):
    def test_visual_default_is_existing_frozen_release(self):
        path = default_release("visual", "/repo")
        self.assertEqual(path, __import__("pathlib").Path("/repo/stage2/releases/accepted_v4_20261105_990b9b48"))

    def test_visual_asr_default_is_separate_future_release(self):
        path = default_release("visual_asr", "/repo", "/cluster/data")
        self.assertEqual(path, __import__("pathlib").Path("/cluster/data/cluster_data/stage2/releases/accepted_av_asr_v1"))

    @patch.dict("os.environ", {}, clear=True)
    def test_bundled_fallback_and_explicit_external_precedence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repo, data = root / "repo", root / "data"
            bundled = repo / "stage2/releases/accepted_av_asr_v1"
            bundled.mkdir(parents=True)
            self.assertEqual(default_release("visual_asr", repo, data), bundled)
            external = data / "cluster_data/stage2/releases/accepted_av_asr_v1"
            external.mkdir(parents=True)
            self.assertEqual(default_release("visual_asr", repo, data), external)
            with patch.dict("os.environ", {"VISUAL_ASR_DATASET_RELEASE": str(root / "explicit")}):
                self.assertEqual(default_release("visual_asr", repo, data), root / "explicit")

    def test_visual_release_cannot_be_used_as_visual_asr(self):
        manifest = {"schema": "stage2_accepted_release_v1", "training_ready": True,
                    "full_dataset_complete": True}
        with self.assertRaisesRegex(ValueError, "frozen release declaring"):
            validate_dataset_version(manifest, "visual_asr")

    def test_av_asr_release_cannot_be_used_as_visual(self):
        manifest = {"schema": "stage2_accepted_release_v1", "training_ready": True,
                    "full_dataset_complete": True,
                    "label_modalities": ["video", "audio_via_window_asr", "asr"]}
        with self.assertRaisesRegex(ValueError, "cannot use an AV\+ASR"):
            validate_dataset_version(manifest, "visual")

    def test_av_asr_version_requires_all_three_modalities(self):
        manifest = {"schema": "stage2_accepted_release_v1", "training_ready": True,
                    "full_dataset_complete": True,
                    "label_modalities": ["video", "audio_via_window_asr", "asr"]}
        validate_dataset_version(manifest, "visual_asr")

    def test_candidates_are_not_trainable(self):
        manifest = {"schema": "stage2_accepted_release_v1", "training_ready": False,
                    "full_dataset_complete": False,
                    "label_modalities": ["video", "audio_via_window_asr", "asr"]}
        with self.assertRaisesRegex(ValueError, "complete frozen release"):
            validate_dataset_version(manifest, "visual_asr")


if __name__ == "__main__":
    unittest.main()
