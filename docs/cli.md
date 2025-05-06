# Command Line Interface

AIF-Gen is intended to be primarily used as a command line tool:

```console
foo@bar:~$ aif --help

          / _ | /  _/ __/ / ___/ __/ |/ /
         / __ |_/ // _/  / (_ / _//    /
        /_/ |_/___/_/    \___/___/_/|_/

A tool for generating synthetic continual RLHF datasets.

Usage: aif [OPTIONS] COMMAND [ARGS]...

Options:
  --log_file FILE  Optional log file to use.  [default: aif_gen.log]
  --help           Show this message and exit.

Commands:
  generate   Generate a new ContinualAlignmentDataset.
  merge      Merge a set of ContinualAlignmentDatasets.
  preview    Preview a ContinualAlignmentDataset.
  sample     Downsample a ContinualAlignmentDataset.
  transform  Transform a ContinualAlignmentDataset.
  validate   Validate a ContinualAlignmentDataset.
```

::: aif_gen.cli.commands.generate.generate
::: aif_gen.cli.commands.merge.merge
::: aif_gen.cli.commands.preview.preview
::: aif_gen.cli.commands.sample.sample
::: aif_gen.cli.commands.transform
::: aif_gen.cli.commands.validate.validate
