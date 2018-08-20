function do1round()
{
    # $1: job name
    # $2: seed
    # $3: job round number
    # $4: conv1 kernel
    # $5: conv2 kernel
    # $6: config.py
    echo "##### job: "${1}" | seed: "${2}" | round: "$3" #####"


    if [ ! -d ${1} ]; then
        mkdir ${1}
    fi
    if [ ! -d ${1}"/"${2} ]; then
        mkdir ${1}"/"${2}
    fi
    if [ ! -d ${1}"/"${2}"/"${3} ]; then
        mkdir ${1}"/"${2}"/"${3}
    fi

    echo "train_path = 'data/"${1}"_train_"${2}".txt'" > ${6} # 1
    echo "test_path = 'data/"${1}"_test_"${2}".txt' " >> ${6} # 2
    echo "result_path = 'work/"${1}"_pred_"${2}"_"${3}".txt'" >> ${6} # 3
    echo "log_path = 'work/"${1}"_"${2}"_"${3}".log'" >> ${6} # 4
    echo "save_prefix = 'work/"${1}"/"${2}"/"${3}"'" >> ${6} # 5
    echo "pretrain_path = ''" >> ${6} # 6
    echo "dict_path = ''" >> ${6} # 7

    echo "" >> ${6} 
    echo "conv1_kernel = "${4} >> ${6} # 8
    echo "conv2_kernel = "${5} >> ${6} # 9

    # these params won't change for HILIC data:
    echo "min_rt = 9" >> ${6}
    echo "max_rt = 47" >> ${6}
    echo "time_scale = 1" >> ${6}
    echo "max_length = 48" >> ${6}

    cd ..
    /root/miniconda3/bin/python capsule_network_emb.py
    cd work
}

function ensemble1sed
{
    # scale = argv[1]
    # round1dir = argv[2] # 'work/dia/59/1/'
    # conv1 = argv[3]
    # round2dir = argv[4] # 'work/dia/59/2/'
    # conv2 = argv[5]
    # round3dir = argv[6] # 'work/dia/59/3/'
    # conv3 = argv[7]
    # result_ensemble = argv[8]
    round1dir="work/"${1}"/"${2}"/"${3}"/"
    conv1=${4}
    round2dir="work/"${1}"/"${2}"/"${5}"/"
    conv2=${6}
    round3dir="work/"${1}"/"${2}"/"${7}"/"
    conv3=${8}
    result="work/"${1}"_notrans_"${2}"_ensemble.txt" # Note thies mark
    cd ..
    /root/miniconda3/bin/python ensemble_emb.py ${9} $round1dir $conv1 $round2dir $conv2 $round3dir $conv3 $result ${10} > "work/"${1}"_notrans_"${2}"_ensemble.log"
    cd work
}

echo "job no.1/10"
# do1round "LUNA_SILICA" 42 1 8 8 "../config.py"
do1round "LUNA_SILICA" 42 2 10 10 "../config.py"
do1round "LUNA_SILICA" 42 3 12 12 "../config.py"
echo "ensemble 3 rounds, 5 epochs each"
ensemble1sed "LUNA_SILICA" 42 '1' 8 '2' 10 '3' 12 9 47 
echo -e "done\n"
