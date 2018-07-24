 #!/bin/sh
PDF_LOC="pdf/*"
PNG_LOC="png/"


for f in $PDF_LOC; do mv "$f" "${f// /_}"; done
exit;
rm -r $PNG_LOC/*
for d in $PDF_LOC; do
    bname="$(basename $d)"
    filename="${bname%.*}"
    echo "Converting $bname ..."
    convert -density 300 $d -quality 100 -alpha off "$PNG_LOC/$filename.png"
done