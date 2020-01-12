python run.py \
--model "scheduled-sampling" \
--name "ss_max" \
--dataset "OpenSubtitles" \
--dataid "../cotk_data/iwslt14" \
--wvid "../cotk_data/glove.6B.300d.txt" \
--lr 0.0001 \
--eh_size 175 \
--dh_size 175 \
--droprate 0.2 \
--device ${1:-0} \
--decode_mode "samplek" \
--epoch 35 \
--seed 2 \
--mode "test" \
--restore "ss_samplek_best.model"