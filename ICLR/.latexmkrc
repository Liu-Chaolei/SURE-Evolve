# Force XeLaTeX for the Chinese manuscript when Overleaf starts with pdfLaTeX.
$pdf_mode = 5;
$xelatex = 'xelatex -interaction=nonstopmode -synctex=1 %O %S';
