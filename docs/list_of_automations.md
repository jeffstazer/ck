[ [Back to index](README.md) ]

# List of CM automations

<!--
This file is generated automatically - don't edit!
-->

* [repo](#repo) *(Managing CM repositories)*
* [script](#script) *(Making native scripts more portable, interoperable and deterministic)*
* [cache](#cache) *(Caching cross-platform CM scripts)*
* [utils](#utils) *(Accessing various CM utils)*
* [core](#core) *(Accessing some core CM functions)*
* [docker](#docker) *(Managing modular docker containers (under development))*
* [experiment](#experiment) *(Managing and reproducing experiments (under development))*
* [ck](#ck) *(Accessing legacy CK automations)*
* [automation](#automation) *(Managing CM automations)*


## repo


*Managing CM repositories.*


* GitHub repository with CM automations: *cm pull [internal](https://github.com/mlcommons/ck/tree/master/cm/cmind/repo)*
* CM automation code and meta: *[GitHub](https://github.com/mlcommons/ck/tree/master/cm/cmind/repo/automation/repo)*
* CM automation actions:
  * cm **pull** repo   &nbsp;&nbsp;&nbsp;*( [See CM API](https://github.com/mlcommons/ck/tree/master/cm/cmind/repo/automation/repo/module.py#L15) )*
  * cm **search** repo   &nbsp;&nbsp;&nbsp;*( [See CM API](https://github.com/mlcommons/ck/tree/master/cm/cmind/repo/automation/repo/module.py#L93) )*
  * cm **update** repo   &nbsp;&nbsp;&nbsp;*( [See CM API](https://github.com/mlcommons/ck/tree/master/cm/cmind/repo/automation/repo/module.py#L172) )*
  * cm **delete** repo   &nbsp;&nbsp;&nbsp;*( [See CM API](https://github.com/mlcommons/ck/tree/master/cm/cmind/repo/automation/repo/module.py#L209) )*
  * cm **init** repo   &nbsp;&nbsp;&nbsp;*( [See CM API](https://github.com/mlcommons/ck/tree/master/cm/cmind/repo/automation/repo/module.py#L262) )*
  * cm **add** repo   &nbsp;&nbsp;&nbsp;*( [See CM API](https://github.com/mlcommons/ck/tree/master/cm/cmind/repo/automation/repo/module.py#L381) )*
  * cm **pack** repo   &nbsp;&nbsp;&nbsp;*( [See CM API](https://github.com/mlcommons/ck/tree/master/cm/cmind/repo/automation/repo/module.py#L389) )*
  * cm **unpack** repo   &nbsp;&nbsp;&nbsp;*( [See CM API](https://github.com/mlcommons/ck/tree/master/cm/cmind/repo/automation/repo/module.py#L459) )*
  * cm **import_ck_to_cm** repo   &nbsp;&nbsp;&nbsp;*( [See CM API](https://github.com/mlcommons/ck/tree/master/cm/cmind/repo/automation/repo/module.py#L562) )*
  * cm **convert_ck_to_cm** repo   &nbsp;&nbsp;&nbsp;*( [See CM API](https://github.com/mlcommons/ck/tree/master/cm/cmind/repo/automation/repo/module.py#L613) )*
  * cm **detect** repo   &nbsp;&nbsp;&nbsp;*( [See CM API](https://github.com/mlcommons/ck/tree/master/cm/cmind/repo/automation/repo/module.py#L667) )*


## script


*Making native scripts more portable, interoperable and deterministic.*


* GitHub repository with CM automations: *cm pull [mlcommons@ck](https://github.com/mlcommons/ck/tree/master/cm-mlops)*
* CM automation code and meta: *[GitHub](https://github.com/mlcommons/ck/tree/master/cm-mlops/automation/script)*
* CM automation actions:
  * cm **run** script   &nbsp;&nbsp;&nbsp;*( [See CM API](https://github.com/mlcommons/ck/tree/master/cm-mlops/automation/script/module.py#L72) )*
  * cm **version** script   &nbsp;&nbsp;&nbsp;*( [See CM API](https://github.com/mlcommons/ck/tree/master/cm-mlops/automation/script/module.py#L1552) )*
  * cm **search** script   &nbsp;&nbsp;&nbsp;*( [See CM API](https://github.com/mlcommons/ck/tree/master/cm-mlops/automation/script/module.py#L1580) )*
  * cm **test** script   &nbsp;&nbsp;&nbsp;*( [See CM API](https://github.com/mlcommons/ck/tree/master/cm-mlops/automation/script/module.py#L1658) )*
  * cm **add** script   &nbsp;&nbsp;&nbsp;*( [See CM API](https://github.com/mlcommons/ck/tree/master/cm-mlops/automation/script/module.py#L1723) )*
  * cm **run_native_script** script   &nbsp;&nbsp;&nbsp;*( [See CM API](https://github.com/mlcommons/ck/tree/master/cm-mlops/automation/script/module.py#L2107) )*
  * cm **find_file_in_paths** script   &nbsp;&nbsp;&nbsp;*( [See CM API](https://github.com/mlcommons/ck/tree/master/cm-mlops/automation/script/module.py#L2148) )*
  * cm **detect_version_using_script** script   &nbsp;&nbsp;&nbsp;*( [See CM API](https://github.com/mlcommons/ck/tree/master/cm-mlops/automation/script/module.py#L2362) )*
  * cm **find_artifact** script   &nbsp;&nbsp;&nbsp;*( [See CM API](https://github.com/mlcommons/ck/tree/master/cm-mlops/automation/script/module.py#L2435) )*
  * cm **find_file_deep** script   &nbsp;&nbsp;&nbsp;*( [See CM API](https://github.com/mlcommons/ck/tree/master/cm-mlops/automation/script/module.py#L2593) )*
  * cm **find_file_back** script   &nbsp;&nbsp;&nbsp;*( [See CM API](https://github.com/mlcommons/ck/tree/master/cm-mlops/automation/script/module.py#L2651) )*
  * cm **parse_version** script   &nbsp;&nbsp;&nbsp;*( [See CM API](https://github.com/mlcommons/ck/tree/master/cm-mlops/automation/script/module.py#L2692) )*
  * cm **update_deps** script   &nbsp;&nbsp;&nbsp;*( [See CM API](https://github.com/mlcommons/ck/tree/master/cm-mlops/automation/script/module.py#L2746) )*
  * cm **get_default_path_list** script   &nbsp;&nbsp;&nbsp;*( [See CM API](https://github.com/mlcommons/ck/tree/master/cm-mlops/automation/script/module.py#L2765) )*
  * cm **doc** script   &nbsp;&nbsp;&nbsp;*( [See CM API](https://github.com/mlcommons/ck/tree/master/cm-mlops/automation/script/module.py#L2776) )*


## cache


*Caching cross-platform CM scripts.*


* GitHub repository with CM automations: *cm pull [mlcommons@ck](https://github.com/mlcommons/ck/tree/master/cm-mlops)*
* CM automation code and meta: *[GitHub](https://github.com/mlcommons/ck/tree/master/cm-mlops/automation/cache)*
* CM automation actions:
  * cm **test** cache   &nbsp;&nbsp;&nbsp;*( [See CM API](https://github.com/mlcommons/ck/tree/master/cm-mlops/automation/cache/module.py#L15) )*
  * cm **show** cache   &nbsp;&nbsp;&nbsp;*( [See CM API](https://github.com/mlcommons/ck/tree/master/cm-mlops/automation/cache/module.py#L54) )*


## utils


*Accessing various CM utils.*


* GitHub repository with CM automations: *cm pull [mlcommons@ck](https://github.com/mlcommons/ck/tree/master/cm-mlops)*
* CM automation code and meta: *[GitHub](https://github.com/mlcommons/ck/tree/master/cm-mlops/automation/utils)*
* CM automation actions:
  * cm **test** utils   &nbsp;&nbsp;&nbsp;*( [See CM API](https://github.com/mlcommons/ck/tree/master/cm-mlops/automation/utils/module.py#L15) )*
  * cm **get_host_os_info** utils   &nbsp;&nbsp;&nbsp;*( [See CM API](https://github.com/mlcommons/ck/tree/master/cm-mlops/automation/utils/module.py#L54) )*
  * cm **download_file** utils   &nbsp;&nbsp;&nbsp;*( [See CM API](https://github.com/mlcommons/ck/tree/master/cm-mlops/automation/utils/module.py#L156) )*
  * cm **unzip_file** utils   &nbsp;&nbsp;&nbsp;*( [See CM API](https://github.com/mlcommons/ck/tree/master/cm-mlops/automation/utils/module.py#L255) )*
  * cm **compare_versions** utils   &nbsp;&nbsp;&nbsp;*( [See CM API](https://github.com/mlcommons/ck/tree/master/cm-mlops/automation/utils/module.py#L333) )*
  * cm **json2yaml** utils   &nbsp;&nbsp;&nbsp;*( [See CM API](https://github.com/mlcommons/ck/tree/master/cm-mlops/automation/utils/module.py#L381) )*
  * cm **yaml2json** utils   &nbsp;&nbsp;&nbsp;*( [See CM API](https://github.com/mlcommons/ck/tree/master/cm-mlops/automation/utils/module.py#L419) )*
  * cm **sort_json** utils   &nbsp;&nbsp;&nbsp;*( [See CM API](https://github.com/mlcommons/ck/tree/master/cm-mlops/automation/utils/module.py#L457) )*
  * cm **dos2unix** utils   &nbsp;&nbsp;&nbsp;*( [See CM API](https://github.com/mlcommons/ck/tree/master/cm-mlops/automation/utils/module.py#L494) )*
  * cm **replace_string_in_file** utils   &nbsp;&nbsp;&nbsp;*( [See CM API](https://github.com/mlcommons/ck/tree/master/cm-mlops/automation/utils/module.py#L531) )*
  * cm **create_toc_from_md** utils   &nbsp;&nbsp;&nbsp;*( [See CM API](https://github.com/mlcommons/ck/tree/master/cm-mlops/automation/utils/module.py#L581) )*
  * cm **copy_to_clipboard** utils   &nbsp;&nbsp;&nbsp;*( [See CM API](https://github.com/mlcommons/ck/tree/master/cm-mlops/automation/utils/module.py#L643) )*


## core


*Accessing some core CM functions.*


* GitHub repository with CM automations: *cm pull [internal](https://github.com/mlcommons/ck/tree/master/cm/cmind/repo)*
* CM automation code and meta: *[GitHub](https://github.com/mlcommons/ck/tree/master/cm/cmind/repo/automation/core)*
* CM automation actions:
  * cm **uid** core   &nbsp;&nbsp;&nbsp;*( [See CM API](https://github.com/mlcommons/ck/tree/master/cm/cmind/repo/automation/core/module.py#L22) )*


## docker


*Managing modular docker containers (under development).*


* GitHub repository with CM automations: *cm pull [mlcommons@ck](https://github.com/mlcommons/ck/tree/master/cm-mlops)*
* CM automation code and meta: *[GitHub](https://github.com/mlcommons/ck/tree/master/cm-mlops/automation/docker)*
* CM automation actions:
  * cm **test** docker   &nbsp;&nbsp;&nbsp;*( [See CM API](https://github.com/mlcommons/ck/tree/master/cm-mlops/automation/docker/module.py#L15) )*


## experiment


*Managing and reproducing experiments (under development).*


* GitHub repository with CM automations: *cm pull [mlcommons@ck](https://github.com/mlcommons/ck/tree/master/cm-mlops)*
* CM automation code and meta: *[GitHub](https://github.com/mlcommons/ck/tree/master/cm-mlops/automation/experiment)*
* CM automation actions:
  * cm **test** experiment   &nbsp;&nbsp;&nbsp;*( [See CM API](https://github.com/mlcommons/ck/tree/master/cm-mlops/automation/experiment/module.py#L15) )*
  * cm **run** experiment   &nbsp;&nbsp;&nbsp;*( [See CM API](https://github.com/mlcommons/ck/tree/master/cm-mlops/automation/experiment/module.py#L53) )*
  * cm **replay** experiment   &nbsp;&nbsp;&nbsp;*( [See CM API](https://github.com/mlcommons/ck/tree/master/cm-mlops/automation/experiment/module.py#L155) )*


## ck


*Accessing legacy CK automations.*


* GitHub repository with CM automations: *cm pull [internal](https://github.com/mlcommons/ck/tree/master/cm/cmind/repo)*
* CM automation code and meta: *[GitHub](https://github.com/mlcommons/ck/tree/master/cm/cmind/repo/automation/ck)*
* CM automation actions:
  * cm **any** ck   &nbsp;&nbsp;&nbsp;*( [See CM API](https://github.com/mlcommons/ck/tree/master/cm/cmind/repo/automation/ck/module.py#L15) )*


## automation


*Managing CM automations.*


* GitHub repository with CM automations: *cm pull [internal](https://github.com/mlcommons/ck/tree/master/cm/cmind/repo)*
* CM automation code and meta: *[GitHub](https://github.com/mlcommons/ck/tree/master/cm/cmind/repo/automation/automation)*
* CM automation actions:
  * cm **add** automation   &nbsp;&nbsp;&nbsp;*( [See CM API](https://github.com/mlcommons/ck/tree/master/cm/cmind/repo/automation/automation/module.py#L15) )*
  * cm **doc** automation   &nbsp;&nbsp;&nbsp;*( [See CM API](https://github.com/mlcommons/ck/tree/master/cm/cmind/repo/automation/automation/module.py#L87) )*



<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
