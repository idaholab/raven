\LoadClass[11pt,twoside]{article}
\typeout{Loading MC2013 document class: includes [11pt,twoside]{article}}
\typeout{adapted by R.C. Martineau at INL from A. Sood at LANL, from J. Wagner 
  (ORNL) who obtained the original class file by Jim Warsa, LANL, 16 July 2002}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% bibliography
\RequirePackage{citesort}

%
% rename reference section
\renewcommand{\refname}{REFERENCES}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% page layout stuff
\RequirePackage{setspace}
\RequirePackage{indentfirst}
\RequirePackage{fancyhdr}
\RequirePackage{afterpage}
\RequirePackage{times}

\setlength{\textwidth}{6.5in}
\setlength{\textheight}{9.0in}
\setlength{\marginparsep}{0pt}
\setlength{\marginparwidth}{0pt}
\setlength{\oddsidemargin}{0pt}
\setlength{\evensidemargin}{0pt}
\setlength{\hoffset}{0pt}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% headers and such (fancyhdr)
%
\setlength{\voffset}{0pt}
\setlength{\hoffset}{0.0in}
\setlength{\topmargin}{0pt}
%\setlength{\topmargin}{16pt}
\setlength{\headsep}{16.8pt}
\setlength{\headheight}{14pt}
\setlength{\footskip}{0.4in}

\setlength{\parindent}{0pt}
\addtolength{\parskip}{1.0\baselineskip}

\fancyhf{}
\renewcommand{\headrulewidth}{0pt}
\renewcommand{\footrulewidth}{0pt}
\lhead{\scriptsize{%
\emph{International Conference on Mathematics and Computational Methods 
 Applied to Nuclear Science \& Engineering (M\&C 2013)} \\
Sun Valley, Idaho, USA, May 5-9, 2013, on CD-ROM, American Nuclear
Society, LaGrange Park, IL (2013)%
}}
\lfoot{}
\pagestyle{fancy}

\def\section{\@startsection{section}{1}{\z@}
{1.0ex plus 1ex minus .2ex}{.2ex plus .2ex}{\normalsize\bf\center}}
\def\subsection{\@startsection{subsection}{2}{\z@} 
{1.0ex plus 1ex minus .2ex}{.2ex plus .2ex}{\normalsize\bf}}
\def\subsubsection{\@startsection{subsubsection}{3}{\z@}
{1.0ex plus 1ex minus .2ex}{.2ex plus .2ex}{\normalsize\bf}}

\renewcommand{\thetable}{\@Roman\c@table}

\newcommand{\Section}[1]{\section{\hskip -1em. #1}}
\newcommand{\Subsection}[1]{\subsection{\hskip -1em. #1}}
\newcommand{\Subsubsection}[1]{\subsubsection{\hskip -1em. #1}}

\renewcommand{\thefootnote}{\fnsymbol{footnote}}
\renewcommand{\footnotesize}{\fontsize{9}{10.8pt}\selectfont}
\renewcommand{\small}{\fontsize{10}{12pt}\selectfont}
\renewcommand{\large}{\fontsize{14}{16.8pt}\selectfont}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% borrowed from pagecounting.sty 
% http://www-users.york.ac.uk/~zrs1/Software/pagecounting.sty)

% History:
%  Created 2000/04/07
% -------- Identification
%
%\NeedsTeXFormat{LaTeX2e}
%\ProvidesPackage{pagecounting}[2000/04/07 <Zoe Stephenson>]
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\newcounter{l@stpage}
\newcommand{\storel@st}{\immediate\write\@mainaux{%
\setcounter{l@stpage}{\thepage}%
% also could use: \setcounter{l@stpage}{\arabic{page}}%
% also could use: \setcounter{l@stpage}{\number{page}}%
\addtocounter{l@stpage}{0}}%  Adjust due to page counter post-advance
}
\renewcommand{\thel@stpage}{\arabic{l@stpage}}
\AtEndDocument{\storel@st}
\newcommand{\totalpages}{\thel@stpage}
\newcommand{\lastpage}{\value{l@stpage}}


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% borrowed from caption.sty in the standard LaTeX2e distribution
%%
%% This is file `caption.sty',
%% generated with the docstrip utility.
%%
%% The original source files were:
%%
%% caption.dtx  (with options: `package')
%% 
%% Copyright (C) 1994-95 Harald Axel Sommerfeldt
%% 
%\NeedsTeXFormat{LaTeX2e}[1994/06/01]
%\ProvidesPackage{caption}[1995/04/05 v1.4b caption package (HS)]
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\typeout{Package: caption v1.4b <1995/04/05> (Harald Sommerfeldt)}
\newcommand{\captionsize}{}
\newcommand{\captionfont}{\captionsize}
\newcommand{\captionlabelfont}{}
\newlength{\captionmargin}
\newcommand{\as@normalcaption}[2]{%
  #1 #2\par}
\newcommand{\as@isucaption}[2]{%
  \sbox{\as@captionbox}{#1\space}%
  \addtolength{\as@captionwidth}{-\wd\as@captionbox}%
  \usebox{\as@captionbox}\parbox[t]{\as@captionwidth}{\leavevmode#2}}
\newcommand{\as@centercaption}[2]{%
  \parbox[t]{\as@captionwidth}{{\centering#1 #2\par}}}
\newcommand{\as@annecaption}[2]{%
  \addtolength{\leftskip}{0pt plus 1fil}%
  \addtolength{\rightskip}{0pt plus -1fil}%
  \setlength{\parfillskip}{0pt plus 2fil}%
  #1 #2\par}
\let\as@shortcaption\as@centercaption
\ifx\thesubfigure\undefined
  \newcommand{\as@subcapsize}[1]{}
\else%
  \typeout{\space\space\space\space\space\space\space\space\space
           `subfigure' package detected}
  \let\as@subcapsize\subcapsize
  \renewcommand{\@thesubfigure}{{\captionlabelfont\thesubfigure}\space}
  \renewcommand{\@thesubtable}{{\captionlabelfont\thesubtable}\space}
\fi
\let\as@caption\as@normalcaption
\newsavebox{\as@captionbox}
\newlength{\as@captionwidth}
\newcommand{\as@makecaption}[2]{%
  \setlength{\leftskip}{\captionmargin}%
  \setlength{\rightskip}{\captionmargin}%
  \addtolength{\as@captionwidth}{-2\captionmargin}%
  \captionfont%
  \sbox{\as@captionbox}{{\captionlabelfont \textbf{#1.}} #2}%
  \ifdim \wd\as@captionbox >\as@captionwidth
    \as@caption{{\captionlabelfont \textbf{#1.}}}{#2}%
  \else%
    \as@shortcaption{{\captionlabelfont \textbf{#1.}}}{#2}%
  \fi}
\renewcommand{\@makecaption}[2]{%
  \vskip\abovecaptionskip%
  \setlength{\as@captionwidth}{\linewidth}%
  \as@makecaption{#1}{#2}%
  \vskip\belowcaptionskip}
\ifx\@makerotcaption\undefined
\else
  \typeout{\space\space\space\space\space\space\space\space\space
           `rotating' package detected}
  \renewcommand{\@makerotcaption}[2]{%
    \captionfont%
    \sbox{\as@captionbox}{{\captionlabelfont \textbf{#1.}} #2}%
    \ifdim \wd\as@captionbox > .8\vsize
      \rotatebox{90}{%
        \setlength{\as@captionwidth}{.8\textheight}%
        \begin{minipage}{\as@captionwidth}%
          \as@caption{{\captionlabelfont \textbf{#1.}}}{#2}%
        \end{minipage}}\par
    \else%
      \rotatebox{90}{\usebox{\as@captionbox}}%
    \fi
    \hspace{12pt}}
\fi
\ifx\floatc@plain\undefined
\else
  \typeout{\space\space\space\space\space\space\space\space\space
           `float' package detected}
  \renewcommand\floatc@plain[2]{%
    \setlength{\as@captionwidth}{\linewidth}%
    \as@makecaption{#1}{#2}}
  \ifx\as@ruled\undefined
  \else
    \renewcommand\floatc@ruled[2]{%
      \setlength{\as@captionwidth}{\linewidth}%
      \captionfont%
      \as@caption{{\captionlabelfont \textbf{#1.}}}{#2}}
  \fi
\fi
\endinput
%%
%% End of file `caption.sty'.
