#for i in {0..4}
#do
#    for j in {0..4}
#    do
#        export CUDA_VISIBLE_DEVICES=0
#        python /home/timongurcke/data-tmp/appropriateness-style-transfer/src/annotation-study-appropriateness-prediction/multilabel_debertav3.py --fold ${i} --repeat ${j} --output /home/timongurcke/data-tmp/appropriateness-style-transfer/data/models/multilabel-debertav3-/fold${i} --input /home/timongurcke/data-tmp/appropriateness-style-transfer/data/appropriateness-corpus/appropriateness_corpus_full_w_folds.csv --issue
#    done
#done

for i in {0..4}
do
    for j in {0..4}
    do
        export CUDA_VISIBLE_DEVICES=0
        python /home/timongurcke/data-tmp/appropriateness-style-transfer/src/annotation-study-appropriateness-prediction/multilabel_debertav3.py --fold ${i} --repeat ${j} --output /home/timongurcke/data-tmp/appropriateness-style-transfer/data/models/multilabel-debertav3-conservative-wo-issue/fold${i} --input /home/timongurcke/data-tmp/appropriateness-style-transfer/data/appropriateness-corpus/appropriateness_corpus_conservative_w_folds.csv 
    done
done

for i in {0..4}
do
    for j in {0..4}
    do
        export CUDA_VISIBLE_DEVICES=0
        python /home/timongurcke/data-tmp/appropriateness-style-transfer/src/annotation-study-appropriateness-prediction/multilabel_debertav3.py --fold ${i} --repeat ${j} --output /home/timongurcke/data-tmp/appropriateness-style-transfer/data/models/multilabel-debertav3-conservative-shuffle/fold${i} --input /home/timongurcke/data-tmp/appropriateness-style-transfer/data/appropriateness-corpus/appropriateness_corpus_conservative_w_folds.csv --issue --shuffle 
    done
done

