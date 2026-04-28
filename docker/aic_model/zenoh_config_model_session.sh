#!/bin/bash

HERE="$(dirname "${BASH_SOURCE[0]}")"
ZENOH_CONFIG_OVERRIDE='transport/auth/usrpwd/user="model"'
ZENOH_CONFIG_OVERRIDE+=';transport/auth/usrpwd/password="CHANGE_IN_PROD"'
ZENOH_CONFIG_OVERRIDE+=';transport/shared_memory/enabled=false'
export ZENOH_CONFIG_OVERRIDE
export ZENOH_SESSION_CONFIG_URI="$(readlink -f "$HERE"/../aic_eval/aic_zenoh_config.json5)"
