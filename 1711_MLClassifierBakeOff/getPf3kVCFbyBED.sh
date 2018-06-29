#!/bin/bash

FOLDER=/seq/plasmodium/data/pf3k/5.1/vcf

while getopts "P:b:" opt; do
  case $opt in
    P) FOLDER=${OPTARG} ;;
    b) bedfile=$OPTARG ;;
  esac
 done

#get header
tabix -H ${FOLDER}/SNP_INDEL_Pf3D7_01_v3.combined.filtered.vcf.gz 

while read chr st en label
do
    
    echo "${chr}:${st}-${en}" >>/dev/stderr
    
    tabix ${FOLDER}/SNP_INDEL_${chr}.combined.filtered.vcf.gz    "${chr}:${st}-${en}"
done < "$bedfile"
