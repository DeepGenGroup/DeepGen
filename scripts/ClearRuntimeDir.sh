#!/bin/bash
temp=$(dirname "$0")
cd ${temp}/..
mydir=`pwd`
echo $mydir ; cd ${mydir} 
rm -rf ${mydir}/_dump
mkdir  ${mydir}/_dump
rm -rf ${mydir}/_cluster_run
mkdir  ${mydir}/_cluster_run
rm -rf ${mydir}/_cache
mkdir  ${mydir}/_cache
rm -rf ${mydir}/_override
mkdir  ${mydir}/_override
rm -rf ${mydir}/_pkls
mkdir  ${mydir}/_pkls
rm -rf ${mydir}/_tmp
mkdir  ${mydir}/_tmp
