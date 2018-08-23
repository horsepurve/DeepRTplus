function do1round()
{
    # $1: job name
    # $2: seed
    # $3: job round number
    # $4: conv1 kernel
    # $5: conv2 kernel
    # $6: config.py
    # $7: pretrain_path
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
    echo "pretrain_path = '"${7}"'" >> ${6} # 6 # Note this difference
    echo "dict_path = 'data/mod_train_2.txt'" >> ${6} # 7

    echo "" >> ${6}
    echo "conv1_kernel = "${4} >> ${6} # 8
    echo "conv2_kernel = "${5} >> ${6} # 9

    # these params won't change for mod data:
    echo "min_rt = 0" >> ${6}
    echo "max_rt = 110" >> ${6}
    echo "time_scale = 60" >> ${6}
    echo "max_length = 50" >> ${6}

    cd ..
    python capsule_network_emb.py
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
    result="work/"${1}"_trans_"${2}"_ensemble.txt" # Note this mark
    cd ..
    python ensemble_emb.py $9 $round1dir $conv1 $round2dir $conv2 $round3dir $conv3 $result ${10} > "work/"${1}"_trans_"${2}"_ensemble.log"
    cd work
}

echo "job no.1/10"
do1round "mod" 2 'trans1' 8 8 "../config.py" "dia_all_epo20_dim24_conv8_filled.pt"
do1round "mod" 2 'trans2' 10 10 "../config.py" "dia_all_epo20_dim24_conv10_filled.pt"
do1round "mod" 2 'trans3' 12 12 "../config.py" "dia_all_epo20_dim24_conv12_filled.pt"
echo "ensemble 3 rounds, 5 epochs each"
ensemble1sed "mod" 2 'trans1' 8 'trans2' 10 'trans3' 12 0 110
echo -e "done\n"

echo "job no.2/10"
do1round "mod" 23 'trans1' 8 8 "../config.py" "dia_all_epo20_dim24_conv8_filled.pt"
do1round "mod" 23 'trans2' 10 10 "../config.py" "dia_all_epo20_dim24_conv10_filled.pt"
do1round "mod" 23 'trans3' 12 12 "../config.py" "dia_all_epo20_dim24_conv12_filled.pt"
echo "ensemble 3 rounds, 5 epochs each"
ensemble1sed "mod" 23 'trans1' 8 'trans2' 10 'trans3' 12 0 110
echo -e "done\n"

echo "job no.3/10"
do1round "mod" 59 'trans1' 8 8 "../config.py" "dia_all_epo20_dim24_conv8_filled.pt"
do1round "mod" 59 'trans2' 10 10 "../config.py" "dia_all_epo20_dim24_conv10_filled.pt"
do1round "mod" 59 'trans3' 12 12 "../config.py" "dia_all_epo20_dim24_conv12_filled.pt"
echo "ensemble 3 rounds, 5 epochs each"
ensemble1sed "mod" 59 'trans1' 8 'trans2' 10 'trans3' 12 0 110
echo -e "done\n"

echo "job no.4/10"
do1round "mod" 97 'trans1' 8 8 "../config.py" "dia_all_epo20_dim24_conv8_filled.pt"
do1round "mod" 97 'trans2' 10 10 "../config.py" "dia_all_epo20_dim24_conv10_filled.pt"
do1round "mod" 97 'trans3' 12 12 "../config.py" "dia_all_epo20_dim24_conv12_filled.pt"
echo "ensemble 3 rounds, 5 epochs each"
ensemble1sed "mod" 97 'trans1' 8 'trans2' 10 'trans3' 12 0 110
echo -e "done\n"

echo "job no.5/10"
do1round "mod" 137 'trans1' 8 8 "../config.py" "dia_all_epo20_dim24_conv8_filled.pt"
do1round "mod" 137 'trans2' 10 10 "../config.py" "dia_all_epo20_dim24_conv10_filled.pt"
do1round "mod" 137 'trans3' 12 12 "../config.py" "dia_all_epo20_dim24_conv12_filled.pt"
echo "ensemble 3 rounds, 5 epochs each"
ensemble1sed "mod" 137 'trans1' 8 'trans2' 10 'trans3' 12 0 110
echo -e "done\n"

echo "job no.6/10"
do1round "mod" 179 'trans1' 8 8 "../config.py" "dia_all_epo20_dim24_conv8_filled.pt"
do1round "mod" 179 'trans2' 10 10 "../config.py" "dia_all_epo20_dim24_conv10_filled.pt"
do1round "mod" 179 'trans3' 12 12 "../config.py" "dia_all_epo20_dim24_conv12_filled.pt"
echo "ensemble 3 rounds, 5 epochs each"
ensemble1sed "mod" 179 'trans1' 8 'trans2' 10 'trans3' 12 0 110
echo -e "done\n"

echo "job no.7/10"
do1round "mod" 227 'trans1' 8 8 "../config.py" "dia_all_epo20_dim24_conv8_filled.pt"
do1round "mod" 227 'trans2' 10 10 "../config.py" "dia_all_epo20_dim24_conv10_filled.pt"
do1round "mod" 227 'trans3' 12 12 "../config.py" "dia_all_epo20_dim24_conv12_filled.pt"
echo "ensemble 3 rounds, 5 epochs each"
ensemble1sed "mod" 227 'trans1' 8 'trans2' 10 'trans3' 12 0 110
echo -e "done\n"

echo "job no.8/10"
do1round "mod" 269 'trans1' 8 8 "../config.py" "dia_all_epo20_dim24_conv8_filled.pt"
do1round "mod" 269 'trans2' 10 10 "../config.py" "dia_all_epo20_dim24_conv10_filled.pt"
do1round "mod" 269 'trans3' 12 12 "../config.py" "dia_all_epo20_dim24_conv12_filled.pt"
echo "ensemble 3 rounds, 5 epochs each"
ensemble1sed "mod" 269 'trans1' 8 'trans2' 10 'trans3' 12 0 110
echo -e "done\n"

echo "job no.9/10"
do1round "mod" 313 'trans1' 8 8 "../config.py" "dia_all_epo20_dim24_conv8_filled.pt"
do1round "mod" 313 'trans2' 10 10 "../config.py" "dia_all_epo20_dim24_conv10_filled.pt"
do1round "mod" 313 'trans3' 12 12 "../config.py" "dia_all_epo20_dim24_conv12_filled.pt"
echo "ensemble 3 rounds, 5 epochs each"
ensemble1sed "mod" 313 'trans1' 8 'trans2' 10 'trans3' 12 0 110
echo -e "done\n"

echo "job no.10/10"
do1round "mod" 367 'trans1' 8 8 "../config.py" "dia_all_epo20_dim24_conv8_filled.pt"
do1round "mod" 367 'trans2' 10 10 "../config.py" "dia_all_epo20_dim24_conv10_filled.pt"
do1round "mod" 367 'trans3' 12 12 "../config.py" "dia_all_epo20_dim24_conv12_filled.pt"
echo "ensemble 3 rounds, 5 epochs each"
ensemble1sed "mod" 367 'trans1' 8 'trans2' 10 'trans3' 12 0 110
echo -e "done\n"