# Cheatsheet

## Page Layout
For using landscape for a single page:

```latex
\begin{landscape}
    <PAGE CONTENT>
\end{landscape}
```


## Headings
For using math mode in headings

```latex
\section{Title with math \texorpdfstring{$\sigma$}{[sigma]}}
```

If you have a very long heading which you want to shorten in the ToC, use the following:
```latex
\section[Short title shown in the TOC]{A very long heading}
```

You can hide a section in the ToC:

```latex
\tochide\section{Hidden section}
```

## Tables

```latex
\begin{table}
    \caption{Even better looking table using booktabs}
    \centering
    \label{table:good_table}
    \begin{tabular}{l c c c c}
        \toprule
        \multirow{2}{*}{Dental measurement} & \multicolumn{2}{c}{Species I} & \multicolumn{2}{c}{Species II} \\ 
        \cmidrule{2-5}
        & mean & SD  & mean & SD  \\ 
        \midrule
        I1MD & 6.23 & 0.91 & 5.2  & 0.7  \\

        I1LL & 7.48 & 0.56 & 8.7  & 0.71 \\

        I2MD & 3.99 & 0.63 & 4.22 & 0.54 \\

        I2LL & 6.81 & 0.02 & 6.66 & 0.01 \\

        CMD & 13.47 & 0.09 & 10.55 & 0.05 \\

        CBL & 11.88 & 0.05 & 13.11 & 0.04\\ 
        \bottomrule
    \end{tabular}
\end{table}
```

## Figures

Normal figures should look like

```latex
\begin{figure}[htbp!] 
    \centering    
    \includegraphics[width=1.0\textwidth]{minion}
    \caption{figure caption}
    \label{fig:label}
\end{figure}
```

For figures showing subplots use
```latex
\begin{figure}
  \centering
  \begin{subfigure}[b]{0.3\textwidth}
    \includegraphics[width=\textwidth]{TomandJerry}
    \caption{Tom and Jerry}
    \label{fig:TomJerry}   
  \end{subfigure}             
  \begin{subfigure}[b]{0.3\textwidth}
    \includegraphics[width=\textwidth]{WallE}
    \caption{Wall-E}
    \label{fig:WallE}
  \end{subfigure}             
  \begin{subfigure}[b]{0.3\textwidth}
    \includegraphics[width=\textwidth]{minion}
    \caption{Minions}
    \label{fig:Minnion}
  \end{subfigure}
  \caption{Best Animations}
  \label{fig:animations}
\end{figure}
```


As for headings, you can shorten the figure captions shown in the ToF using
```latex
\caption[short caption shown in ToF]{Very long caption}
```

## Code

Non-specific code goes in

```latex
\begin{verbatim}
mount -t iso9660 -o ro,loop,noauto /your/texlive####.iso /mnt
\end{verbatim}
```

## Nomenclature

Adding entires to the nomenclature

```latex
\nomenclature[z-cif]{$CIF$}{Cauchy's Integral Formula}  % first letter Z is for Acronyms 
\nomenclature[a-F]{$F$}{complex function}  % first letter A is for Roman symbols
\nomenclature[g-p]{$\pi$}{ $\simeq 3.14\ldots$}  % first letter G is for Greek Symbols
\nomenclature[g-i]{$\iota$}{unit imaginary number $\sqrt{-1}$}  % first letter G is for Greek Symbols
\nomenclature[g-g]{$\gamma$}{a simply closed curve on a complex plane}  % first letter G is for Greek Symbols
\nomenclature[x-i]{$\oint_\gamma$}{integration around a curve $\gamma$} % first letter X is for Other Symbols
\nomenclature[r-j]{$j$}{superscript index}  % first letter R is for superscripts
\nomenclature[s-0]{$0$}{subscript index}
```

## Index
Adding an entry to the index
```
{\em \LaTeX{} class file}\index{\LaTeX{} class file@LaTeX class file}
```
