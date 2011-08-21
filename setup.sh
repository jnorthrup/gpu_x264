#!/bin/bash
exit_install() {
	echo
	echo "After program was build successfull, run demo.sh"
	exit 0
}

mac_install() {	
	if [ -f /System/Library/Frameworks/OpenCL.framework/Versions/A/Headers/opencl.h ]; then
		:
	else
		echo "No valid OpenCL installation found"
		echo "Check for Xcode Development Tools Installation"
		exit
	fi

	./configure --host=x86_64-apple-darwin --enable-opencl --enable-debug
	exit_install
}

linux_install() {
	echo "Enter path to CL/cl.h header file and press [ENTER]: "
	read linux_path
	./configure --enable-debug --enable-opencl --linux-opencl-dir=$linux_path
	exit_install
}

echo; echo "OpenCL powered x264 Installation"
echo; echo "################################"
while : # Loop forever
do
echo
echo "Select Operating System:"
echo "[1] MacOS X Snow Leopard 10.6"
echo "[2] Linux with OpenCL support"
echo "[X] Exit"
read Keypress

case "$Keypress" in
  [1]   ) mac_install;;
  [2]   ) linux_install;;
  [X]   ) exit 0;;
  *     ) echo "Selection not valid, make your choice once more";;
esac      #  Allows ranges of characters in [square brackets],
          #+ or POSIX ranges in [[double square brackets.
done

