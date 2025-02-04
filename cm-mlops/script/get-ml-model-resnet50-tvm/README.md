<details>
<summary>Click here to see the table of contents.</summary>

* [Description](#description)
* [Information](#information)
* [Usage](#usage)
  * [ CM installation](#cm-installation)
  * [ CM script automation help](#cm-script-automation-help)
  * [ CM CLI](#cm-cli)
  * [ CM Python API](#cm-python-api)
  * [ CM modular Docker container](#cm-modular-docker-container)
* [Customization](#customization)
  * [ Variations](#variations)
  * [ Default environment](#default-environment)
* [Script workflow, dependencies and native scripts](#script-workflow-dependencies-and-native-scripts)
* [Script output](#script-output)
* [New environment keys (filter)](#new-environment-keys-(filter))
* [New environment keys auto-detected from customize](#new-environment-keys-auto-detected-from-customize)
* [Maintainers](#maintainers)

</details>

*Note that this README is automatically generated - don't edit! See [more info](README-extra.md).*

### Description


See [more info](README-extra.md).

#### Information

* Category: *ML/AI models.*
* CM GitHub repository: *[mlcommons@ck](https://github.com/mlcommons/ck/tree/master/cm-mlops)*
* GitHub directory for this script: *[GitHub](https://github.com/mlcommons/ck/tree/master/cm-mlops/script/get-ml-model-resnet50-tvm)*
* CM meta description for this script: *[_cm.json](_cm.json)*
* CM "database" tags to find this script: *get,ml-model,ml-model-tvm,tvm-model,resnet50,ml-model-resnet50,image-classification*
* Output cached?: *True*
___
### Usage

#### CM installation
[Guide](https://github.com/mlcommons/ck/blob/master/docs/installation.md)

#### CM script automation help
```cm run script --help```

#### CM CLI
`cm run script --tags=get,ml-model,ml-model-tvm,tvm-model,resnet50,ml-model-resnet50,image-classification(,variations from below) (flags from below)`

*or*

`cm run script "get ml-model ml-model-tvm tvm-model resnet50 ml-model-resnet50 image-classification (variations from below)" (flags from below)`

*or*

`cm run script c1b7b656b6224307`

#### CM Python API

<details>
<summary>Click here to expand this section.</summary>

```python

import cmind

r = cmind.access({'action':'run'
                  'automation':'script',
                  'tags':'get,ml-model,ml-model-tvm,tvm-model,resnet50,ml-model-resnet50,image-classification'
                  'out':'con',
                  ...
                  (other input keys for this script)
                  ...
                 })

if r['return']>0:
    print (r['error'])

```

</details>

#### CM modular Docker container
*TBD*
___
### Customization


#### Variations

  * Group "**batchsize**"
    <details>
    <summary>Click here to expand this section.</summary>

    * `_bs.1`
      - Environment variables:
        - *CM_ML_MODEL_MAX_BATCH_SIZE*: `1`
      - Workflow:
    * `_bs.16`
      - Environment variables:
        - *CM_ML_MODEL_MAX_BATCH_SIZE*: `16`
      - Workflow:
    * `_bs.2`
      - Environment variables:
        - *CM_ML_MODEL_MAX_BATCH_SIZE*: `2`
      - Workflow:
    * `_bs.32`
      - Environment variables:
        - *CM_ML_MODEL_MAX_BATCH_SIZE*: `32`
      - Workflow:
    * `_bs.4`
      - Environment variables:
        - *CM_ML_MODEL_MAX_BATCH_SIZE*: `4`
      - Workflow:
    * `_bs.64`
      - Environment variables:
        - *CM_ML_MODEL_MAX_BATCH_SIZE*: `64`
      - Workflow:
    * **`_bs.8`** (default)
      - Environment variables:
        - *CM_ML_MODEL_MAX_BATCH_SIZE*: `8`
      - Workflow:

    </details>


  * Group "**framework**"
    <details>
    <summary>Click here to expand this section.</summary>

    * **`_onnx`** (default)
      - Workflow:
        1. ***Read "deps" on other CM scripts***
           * get,ml-model,raw,resnet50,_onnx
             * CM names: `--adr.['original-model']...`
             - CM script: [get-ml-model-resnet50](https://github.com/mlcommons/ck/tree/master/cm-mlops/script/get-ml-model-resnet50)
    * `_pytorch`
      - Workflow:
        1. ***Read "deps" on other CM scripts***
           * get,ml-model,raw,resnet50,_pytorch
             * CM names: `--adr.['original-model']...`
             - CM script: [get-ml-model-resnet50](https://github.com/mlcommons/ck/tree/master/cm-mlops/script/get-ml-model-resnet50)
    * `_tensorflow`
      - Aliases: `_tf,_tflite`
      - Workflow:
        1. ***Read "deps" on other CM scripts***
           * get,ml-model,raw,resnet50,_tf
             * CM names: `--adr.['original-model']...`
             - CM script: [get-ml-model-resnet50](https://github.com/mlcommons/ck/tree/master/cm-mlops/script/get-ml-model-resnet50)

    </details>


  * Group "**precision**"
    <details>
    <summary>Click here to expand this section.</summary>

    * **`_fp32`** (default)
      - Workflow:
    * `_int8`
      - Workflow:
    * `_uint8`
      - Workflow:

    </details>


#### Default variations

`_bs.8,_fp32,_onnx`
#### Default environment

<details>
<summary>Click here to expand this section.</summary>

These keys can be updated via --env.KEY=VALUE or "env" dictionary in @input.json or using script flags.


</details>

___
### Script workflow, dependencies and native scripts

  1. ***Read "deps" on other CM scripts from [meta](https://github.com/mlcommons/ck/tree/master/cm-mlops/script/get-ml-model-resnet50-tvm/_cm.json)***
     * get,python3
       * CM names: `--adr.['python, python3']...`
       - CM script: [get-python3](https://github.com/mlcommons/ck/tree/master/cm-mlops/script/get-python3)
     * get,tvm
       * CM names: `--adr.['tvm']...`
       - CM script: [get-tvm](https://github.com/mlcommons/ck/tree/master/cm-mlops/script/get-tvm)
  1. ***Run "preprocess" function from [customize.py](https://github.com/mlcommons/ck/tree/master/cm-mlops/script/get-ml-model-resnet50-tvm/customize.py)***
  1. Read "prehook_deps" on other CM scripts from [meta](https://github.com/mlcommons/ck/tree/master/cm-mlops/script/get-ml-model-resnet50-tvm/_cm.json)
  1. ***Run native script if exists***
     * [run.sh](https://github.com/mlcommons/ck/tree/master/cm-mlops/script/get-ml-model-resnet50-tvm/run.sh)
  1. Read "posthook_deps" on other CM scripts from [meta](https://github.com/mlcommons/ck/tree/master/cm-mlops/script/get-ml-model-resnet50-tvm/_cm.json)
  1. ***Run "postrocess" function from [customize.py](https://github.com/mlcommons/ck/tree/master/cm-mlops/script/get-ml-model-resnet50-tvm/customize.py)***
  1. Read "post_deps" on other CM scripts from [meta](https://github.com/mlcommons/ck/tree/master/cm-mlops/script/get-ml-model-resnet50-tvm/_cm.json)
___
### Script output
#### New environment keys (filter)

* **CM_ML_MODEL_***
#### New environment keys auto-detected from customize

* **CM_ML_MODEL_FILE**
* **CM_ML_MODEL_FILE_WITH_PATH**
* **CM_ML_MODEL_FRAMEWORK**
* **CM_ML_MODEL_INPUT_SHAPES**
* **CM_ML_MODEL_ORIGINAL_FILE_WITH_PATH**
* **CM_ML_MODEL_PATH**
___
### Maintainers

* [Open MLCommons taskforce on education and reproducibility](https://github.com/mlcommons/ck/blob/master/docs/mlperf-education-workgroup.md)