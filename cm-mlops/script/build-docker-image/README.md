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
  * [ Script flags mapped to environment](#script-flags-mapped-to-environment)
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

* Category: *Docker automation.*
* CM GitHub repository: *[mlcommons@ck](https://github.com/mlcommons/ck/tree/master/cm-mlops)*
* GitHub directory for this script: *[GitHub](https://github.com/mlcommons/ck/tree/master/cm-mlops/script/build-docker-image)*
* CM meta description for this script: *[_cm.json](_cm.json)*
* CM "database" tags to find this script: *build,docker,image,docker-image,dockerimage*
* Output cached?: *False*
___
### Usage

#### CM installation
[Guide](https://github.com/mlcommons/ck/blob/master/docs/installation.md)

#### CM script automation help
```cm run script --help```

#### CM CLI
`cm run script --tags=build,docker,image,docker-image,dockerimage(,variations from below) (flags from below)`

*or*

`cm run script "build docker image docker-image dockerimage (variations from below)" (flags from below)`

*or*

`cm run script 2c3c4ba2413442e7`

#### CM Python API

<details>
<summary>Click here to expand this section.</summary>

```python

import cmind

r = cmind.access({'action':'run'
                  'automation':'script',
                  'tags':'build,docker,image,docker-image,dockerimage'
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


#### Script flags mapped to environment
<details>
<summary>Click here to expand this section.</summary>

* --**cache**=value --> **CM_DOCKER_CACHE**=value
* --**cm_repo**=value --> **CM_MLOPS_REPO**=value
* --**docker_os**=value --> **CM_DOCKER_OS**=value
* --**docker_os_version**=value --> **CM_DOCKER_OS_VERSION**=value
* --**dockerfile**=value --> **CM_DOCKERFILE_WITH_PATH**=value
* --**gh_token**=value --> **CM_GH_TOKEN**=value
* --**image_name**=value --> **CM_DOCKER_IMAGE_NAME**=value
* --**image_repo**=value --> **CM_DOCKER_IMAGE_REPO**=value
* --**image_tag**=value --> **CM_DOCKER_IMAGE_TAG**=value
* --**post_run_cmds**=value --> **CM_DOCKER_POST_RUN_COMMANDS**=value
* --**pre_run_cmds**=value --> **CM_DOCKER_PRE_RUN_COMMANDS**=value
* --**real_run**=value --> **CM_REAL_RUN**=value
* --**script_tags**=value --> **CM_DOCKER_RUN_SCRIPT_TAGS**=value

**Above CLI flags can be used in the Python CM API as follows:**

```python
r=cm.access({... , "cache":...}
```

</details>

#### Default environment

<details>
<summary>Click here to expand this section.</summary>

These keys can be updated via --env.KEY=VALUE or "env" dictionary in @input.json or using script flags.

* CM_DOCKER_IMAGE_REPO: **local**
* CM_DOCKER_IMAGE_TAG: **latest**

</details>

___
### Script workflow, dependencies and native scripts

  1. Read "deps" on other CM scripts from [meta](https://github.com/mlcommons/ck/tree/master/cm-mlops/script/build-docker-image/_cm.json)
  1. ***Run "preprocess" function from [customize.py](https://github.com/mlcommons/ck/tree/master/cm-mlops/script/build-docker-image/customize.py)***
  1. ***Read "prehook_deps" on other CM scripts from [meta](https://github.com/mlcommons/ck/tree/master/cm-mlops/script/build-docker-image/_cm.json)***
     * build,dockerfile
       * `if (CM_BUILD_DOCKERFILE in ['yes', '1'])`
       - CM script: [build-dockerfile](https://github.com/mlcommons/ck/tree/master/cm-mlops/script/build-dockerfile)
  1. ***Run native script if exists***
     * [run.bat](https://github.com/mlcommons/ck/tree/master/cm-mlops/script/build-docker-image/run.bat)
     * [run.sh](https://github.com/mlcommons/ck/tree/master/cm-mlops/script/build-docker-image/run.sh)
  1. Read "posthook_deps" on other CM scripts from [meta](https://github.com/mlcommons/ck/tree/master/cm-mlops/script/build-docker-image/_cm.json)
  1. Run "postrocess" function from customize.py
  1. Read "post_deps" on other CM scripts from [meta](https://github.com/mlcommons/ck/tree/master/cm-mlops/script/build-docker-image/_cm.json)
___
### Script output
#### New environment keys (filter)

* **CM_DOCKER_***
#### New environment keys auto-detected from customize

* **CM_DOCKER_BUILD_ARGS**
* **CM_DOCKER_CACHE_ARG**
* **CM_DOCKER_IMAGE_NAME**
* **CM_DOCKER_IMAGE_REPO**
* **CM_DOCKER_IMAGE_TAG**
___
### Maintainers

* [Open MLCommons taskforce on education and reproducibility](https://github.com/mlcommons/ck/blob/master/docs/mlperf-education-workgroup.md)