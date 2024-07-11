#!/usr/bin/env bash

# Fetch the PSL examples and modify the CLI configuration for these experiments.

readonly BASE_DIR=$(realpath "$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"/..)

readonly PSL_EXAMPLES_DIR="${BASE_DIR}/psl-examples"
readonly PSL_EXAMPLES_REPO='https://github.com/linqs/psl-examples.git'
readonly PSL_EXAMPLES_BRANCH='main'

readonly AVAILABLE_MEM_KB=$(cat /proc/meminfo | grep 'MemTotal' | sed 's/^[^0-9]\+\([0-9]\+\)[^0-9]\+$/\1/')
# Floor by multiples of 5 and then reserve an additional 5 GB.
readonly JAVA_MEM_GB=$((${AVAILABLE_MEM_KB} / 1024 / 1024 / 5 * 5 - 5))
#readonly JAVA_MEM_GB=16

function fetch_psl_examples() {
   if [ -e ${PSL_EXAMPLES_DIR} ]; then
      return
   fi

   echo "Models and data not found, fetching them."
   git clone ${PSL_EXAMPLES_REPO} ${PSL_EXAMPLES_DIR}

   pushd . > /dev/null
      cd "${PSL_EXAMPLES_DIR}"
      git checkout ${PSL_EXAMPLES_BRANCH}
   popd > /dev/null
}

function fetch_jar() {
    # Only make a new out directory if it does not already exist
    [[ -d "${BASE_DIR}/psl_resources" ]] || mkdir -p "${BASE_DIR}/psl_resources"

    # psl 3.0.1
    local snapshotJARPath="$HOME/.m2/repository/org/linqs/psl-cli/3.0.1-SNAPSHOT/psl-cli-3.0.1-SNAPSHOT.jar"
    cp "${snapshotJARPath}" "${BASE_DIR}/psl_resources/psl-cli-3.0.1-SNAPSHOT.jar"
}

# Common to all examples.
function standard_fixes() {
    echo "$BASE_DIR"
    for exampleDir in `find ${PSL_EXAMPLES_DIR} -maxdepth 1 -mindepth 1 -type d -not -name '.*' -not -name '_scripts'`; do
        local baseName=`basename ${exampleDir}`

        pushd . > /dev/null
            cd "${exampleDir}/cli"

            # Increase memory allocation.
            sed -i "s/java -jar/java -Xmx${JAVA_MEM_GB}G -Xms${JAVA_MEM_GB}G -jar/" run.sh

            # cp 3.0.1 snapshot into the cli directory.
            cp ../../../psl_resources/psl-cli-3.0.1-SNAPSHOT.jar ./

            # Deactivate fetch psl step.
            sed -i 's/^\(\s\+\)fetch_psl/\1# fetch_psl/' run.sh

            # Restore the .json file.
            git restore "${baseName}.json"

            # Remove comments from the .json file.
            sed -i 's/^\s\+\/\/.*$//' "${baseName}.json"
            sed -i 's/^\s\+#.*$//' "${baseName}.json"

        popd > /dev/null
    done
}

function main() {
   trap exit SIGINT

   fetch_psl_examples
   fetch_jar
   standard_fixes

   exit 0
}

[[ "${BASH_SOURCE[0]}" == "${0}" ]] && main "$@"
