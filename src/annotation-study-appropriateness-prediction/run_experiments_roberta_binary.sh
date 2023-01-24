for i in {0..4}
do
    for j in {0..4}
    do
        export CUDA_VISIBLE_DEVICES=0
        python /home/timongurcke/data-tmp/appropriateness-style-transfer/src/annotation-study-appropriateness-prediction/binary_roberta.py --fold ${i} --repeat ${j} --output /home/timongurcke/data-tmp/appropriateness-style-transfer/data/models/binary-roberta-majority/fold${i} --input /home/timongurcke/data-tmp/appropriateness-style-transfer/data/appropriateness-corpus/appropriateness_corpus_majority_w_folds.csv --issue
    done
done

