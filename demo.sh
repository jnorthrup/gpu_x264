# !/bin/bash

download_png() {
	echo "Downloading all needed video files from bigbugbunny trailer"
    echo http://media.xiph.org/BBB/BBB-1080-png/big_buck_bunny_{10000..10600}.png |xargs -P10 -n10 wget -c
}

convert_demo() {
	echo; echo; echo "Converting PNG files to source file"
	png2yuv -j big_buck_bunny_%05d.png -f 24 -I p -n 600 -b 10000 > result.yuv
	y4mtoyuv <result.y4m > result.yuv
	rm result.y4m
	mv result.yuv result.y4m
}

convert_video() {
	echo; echo; echo "Encoding video material"
	./x264 --threads 8 -A none --no-cabac --no-deblock --subme 0 --me dia --qp 16 --output out.264 result.y4m
}

check_pngFiles() {
	for i in {10000..10600}
	do
		file="./big_buck_bunny_$i.png"
		if [ -f $file ]; then
			:
		else
			wget http://media.xiph.org/BBB/BBB-1080-png/big_buck_bunny_$i.png
		fi
	done
}

check_files() {
	if [ -f result.y4m ]; then
		convert_video
	else
		convert_demo
		convert_video
	fi
}

check_for_programs() {
	check=0;

	if which wget >/dev/null; then
		check=0;
	else
		echo "wget is missing but is needed for further processing"
		echo "please install wget"
		check=1;
	fi

	if which png2yuv >/dev/null; then
		check=0;
	else
		echo "png2yuv is missing but is needed for further processing"
		echo "please install MP4Tool"
		check=1;
	fi
	
	if which y4mtoyuv >/dev/null; then
		check=0;
	else
		echo "y4mtoyuv is missing but is needed for further processing"
		echo "please install MP4Tool"
		check=1;
	fi
	
	echo "check: $check"
	
	if [ $check -eq 1 ]; then
		exit
	fi
}

#check_for_programs
#check_pngFiles
check_files


echo; echo; echo "Play out.264 with mplayer"
