#! /bin/sh

rsync --progress --delete --recursive dev-reference/* buster:doc/hedge-reference
