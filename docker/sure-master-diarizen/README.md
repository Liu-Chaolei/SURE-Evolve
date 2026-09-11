# DiariZen Ascend worker

Create a fresh context with `python prepare_context.py --source /path/to/DiariZen --output /shared/$USER/sd-context`.
Build on n01, push to the personal internal Registry, and use the resulting digest in Slurm.
The base image pins the working Torch/Torch-NPU/CANN stack. Its ASR dependencies are inherited,
but DiariZen and the CPU metric virtualenv are installed separately here.

`md-eval-22.pl` is the NIST scorer distributed by
<https://github.com/nryant/dscore/blob/master/scorelib/md-eval-22.pl>, downloaded on 2026-09-11.
Only line endings were normalized to LF for its Perl shebang. It is included so DER
evaluation does not need to download executable code from inside a training job.

External DiariZen source, datasets and model weights are not committed in this directory.
