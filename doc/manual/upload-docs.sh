#! /bin/sh

rsync --progress --delete --recursive build/html/* doc-upload:doc/hedge
