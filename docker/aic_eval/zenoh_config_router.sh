#!/bin/bash

HERE="$(dirname "${BASH_SOURCE[0]}")"
CREDENTIALS="$(readlink -f "$HERE"/credentials.txt)"
ZENOH_CONFIG_OVERRIDE='transport/auth/usrpwd/user="eval"'
ZENOH_CONFIG_OVERRIDE+=';transport/auth/usrpwd/password="CHANGE_IN_PROD"'
ZENOH_CONFIG_OVERRIDE+=';transport/auth/usrpwd/dictionary_file="'"$CREDENTIALS"'"'
ZENOH_CONFIG_OVERRIDE+=';transport/shared_memory/enabled=false'
ZENOH_CONFIG_OVERRIDE+=';mode="router"'
ZENOH_CONFIG_OVERRIDE+=';connect/endpoints=[]'
ZENOH_CONFIG_OVERRIDE+=';listen/endpoints=["tcp/[::]:7447"]'
export ZENOH_CONFIG_OVERRIDE
export ZENOH_ROUTER_CONFIG_URI="$(readlink -f "$HERE"/aic_zenoh_config.json5)"
