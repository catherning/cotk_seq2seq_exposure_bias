python run.py ^
--model "scheduled-sampling" ^
--dataset "OpenSubtitles" ^
--dataid "../cotk_data/iwslt14" ^
--wvid "../cotk_data/glove.6B.300d.txt" ^
--lr 0.0005 ^
--eh_size 175 ^
--dh_size 175 ^
--droprate 0.2 ^
--decode_mode "samplek" ^
--epoch 35

