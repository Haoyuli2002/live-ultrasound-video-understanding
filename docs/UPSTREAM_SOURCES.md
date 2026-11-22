# Upstream source provenance

StreamReady is vendored as source files, including its upstream LICENSE, so a clone of this project contains the Qwen3.5 modifications without requiring an unpublished submodule commit.

| Project | Upstream | Base revision | Local treatment |
| --- | --- | --- | --- |
| StreamReady | https://github.com/sacrcv/StreamReady.git | `a5af28b44e3ac92095980dface1be0d58d442eab` | Vendored with local Qwen3.5, SDPA, memory and training adaptations. |
| LAVIS | https://github.com/salesforce/LAVIS.git | `05cb670f47296522e24e7727e366b4a997d28582` | Pristine reference clone removed; adapted query layers and license retained in stage2/. |
| Dispider | https://github.com/Mark12Ding/Dispider.git | `803f6b4634b3232293e89a38bcce9f5815c33ed4` | Reference clone removed; original analysis preserved in docs/DISPIDER_LOGIC_zh.md. |

Reference clones can be reconstructed with `git clone <upstream> <name>` followed by `git -C <name> checkout <revision>`. They are not runtime dependencies of the current end-to-end path.

The original nested StreamReady Git directory is preserved at `/dss/mcmlscratch/04/ge75vid2/tmp/workspace_cleanup_20261122T075247Z/StreamReady.git`; the tracked patch is also archived beside it. The parent repository now owns the source files; avoid restoring a nested `.git` in a working checkout. No pushes are made to third-party upstream repositories.
