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

    echo "train_path = 'data/"${1}"_train_"${2}".txt'" > ${6}
    echo "test_path = 'data/"${1}"_test_"${2}".txt' " >> ${6}
    echo "result_path = 'work/"${1}"_pred_"${2}"_"${3}".txt'" >> ${6}
    echo "log_path = 'work/"${1}"_"${2}"_"${3}".log'" >> ${6}
    echo "save_prefix = 'work/"${1}"/"${2}"/"${3}"'" >> ${6}
    echo "pretrain_path = ''" >> ${6}
    echo "dict_path = ''" >> ${6}

    echo "" >> ${6}
    echo "conv1_kernel = "${4} >> ${6}
    echo "conv2_kernel = "${5} >> ${6}

    cd ..
    /root/miniconda3/bin/python capsule_network.py
    cd work
}

# echo "job no.1/10"
# do1round "dia" 2 1 10 10 "../config.py"
# do1round "dia" 2 2 13 13 "../config.py"
# do1round "dia" 2 3 16 16 "../config.py"
# echo -e "done\n"

# echo "job no.2/10"
# do1round "dia" 23 1 10 10 "../config.py"
# do1round "dia" 23 2 13 13 "../config.py"
# do1round "dia" 23 3 16 16 "../config.py"
# echo -e "done\n"

# echo "job no.3/10"
# do1round "dia" 59 1 10 10 "../config.py"
# do1round "dia" 59 2 8 8 "../config.py"
# do1round "dia" 59 3 12 12 "../config.py"
# echo -e "done\n"

# echo "job no.4/10"
do1round "dia" 97 1 10 10 "../config.py" # we have run to this <-------
# do1round "dia" 97 2 8 8 "../config.py"
# do1round "dia" 97 3 12 12 "../config.py"
# echo -e "done\n"

# echo "job no.5/10"
# do1round "dia" 137 1 10 10 "../config.py"
# do1round "dia" 137 2 8 8 "../config.py"
# do1round "dia" 137 3 12 12 "../config.py"
# echo -e "done\n"

# echo "job no.6/10"
# do1round "dia" 179 1 10 10 "../config.py"
# do1round "dia" 179 2 8 8 "../config.py"
# do1round "dia" 179 3 12 12 "../config.py"
# echo -e "done\n"

# echo "job no.7/10"
# do1round "dia" 227 1 10 10 "../config.py"
# do1round "dia" 227 2 8 8 "../config.py"
# do1round "dia" 227 3 12 12 "../config.py"
# echo -e "done\n"

# echo "job no.8/10"
# do1round "dia" 269 1 10 10 "../config.py"
# do1round "dia" 269 2 8 8 "../config.py"
# do1round "dia" 269 3 12 12 "../config.py"
# echo -e "done\n"

# echo "job no.9/10"
# do1round "dia" 313 1 10 10 "../config.py"
# do1round "dia" 313 2 8 8 "../config.py"
# do1round "dia" 313 3 12 12 "../config.py"
# echo -e "done\n"

# echo "job no.10/10"
# do1round "dia" 367 1 10 10 "../config.py"
# do1round "dia" 367 2 8 8 "../config.py"
# do1round "dia" 367 3 12 12 "../config.py"
# echo -e "done\n"
