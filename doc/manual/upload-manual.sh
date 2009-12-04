#! /bin/sh

rsync --progress --delete --recursive build/html/* buster:doc/hedge
