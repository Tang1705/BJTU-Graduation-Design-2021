%===================================================

%===================================================

%导入需要的宏包
\RequirePackage{amsmath,amsthm,amsfonts,amssymb,bm,mathrsfs,upgreek}
\RequirePackage{fancyhdr}
\RequirePackage{booktabs}
\RequirePackage{multirow}
\RequirePackage[normalem]{ulem}
\RequirePackage{ragged2e}
\RequirePackage{blindtext}

%依赖的Latex版本
\NeedsTeXFormat{LaTeX2e}[2005/12/01]
%声明本文档为模板类
\ProvidesClass{BJTU-thesis}[2018/4/27]

\LoadClass[zihao=-4,a4paper,oneside,openright,UTF8,space=auto]{ctexbook}

%定义使用者需要填写的标签
\def\BJTU@label@schoolNumber{学校代码:~}
\def\BJTU@label@school{学院名称:~}
\def\BJTU@label@classification{密级:~}
\def\BJTU@label@author{作者姓名:~}
\def\BJTU@label@studentNumber{学~~~~~~号:~}
\def\BJTU@label@advisor{导师姓名:~}
\def\BJTU@label@advisorTitle{职~~~~~~称:~}
\def\BJTU@label@engineeringMasterField{工程硕士专业领域:~}
\def\BJTU@label@degreeLevel{学位级别:~}
\def\BJTU@label@degreeType{学位类别:~}
\def\BJTU@label@major{学科专业:~}
\def\BJTU@label@researchArea{研究方向:~}

%定义上述标签的默认值
\def\BJTU@value@schoolNumber{10004}
\def\BJTU@value@school{XXXX}
\def\BJTU@value@classification{公开}
\def\BJTU@value@author{XXXX}
\def\BJTU@value@studentNumber{XXXX}
\def\BJTU@value@advisor{XXXX}
\def\BJTU@value@advisorTitle{XXXX}
\def\BJTU@value@engineeringMasterField{XXXX}
\def\BJTU@value@degreeLevel{硕士}
\def\BJTU@value@degreeType{XXXX}
\def\BJTU@value@major{XXXX}
\def\BJTU@value@researchArea{XXXX}
\def\BJTU@value@title{中文题目}
\def\BJTU@value@englishtitle{英文题目}

%定义用户填写上述标签对应值的命令,需要用户在主文档自行调用
\newcommand\schooNumber[1]{\def\BJTU@value@schoolNumber{#1}}
\newcommand\school[1]{\def\BJTU@value@school{#1}}
\newcommand\classification[1]{\def\BJTU@value@classification{#1}}
\renewcommand\author[1]{\def\BJTU@value@author{#1}}
\newcommand\studentNumber[1]{\def\BJTU@value@studentNumber{#1}}
\newcommand\advisor[1]{\def\BJTU@value@advisor{#1}}
\newcommand\advisorTitle[1]{\def\BJTU@value@advisorTitle{#1}}
\newcommand\engineeringMasterField[1]{\def\BJTU@value@engineeringMasterField{#1}}
\newcommand\degreeType[1]{\def\BJTU@value@degreeType{#1}}
\newcommand\major[1]{\def\BJTU@value@major{#1}}
\newcommand\researchArea[1]{\def\BJTU@value@researchArea{#1}}
\newcommand\englishtitle[1]{\def\BJTU@value@englishtitle{#1}}
\renewcommand\title[1]{\def\BJTU@value@title{#1}}


%版权授权书
\newcommand{\makeAuthorization}{
	\chapter*{\heiti\zihao{-2}学士论文版权使用授权书}
	\thispagestyle{empty}
	\vspace{10pt}
	本学士论文作者完全了解北京交通大学有关保留、使用学士论文的规定。特授权北京交通大学可以将学士论文的全部或部分内容编入有关数据库进行检索，提供阅览服务，并采用影印、缩印或扫描等复制手段保存、汇编以供查阅和借阅。
	
	\vspace{5pt}
		\begin{center}
			(保密的学位论文在解密后适用本授权说明）
		\end{center}
						
	
	\vspace{72pt}
	学位论文作者签名：~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~指导教师签名：
	
	\vspace{12pt}
	签字日期：~~~~~~年~~~月~~~日~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~签字日期：~~~~~年~~~月~~~日 
}



%重新定义致谢环境
\renewenvironment{thanks}{
	{\centering\chapter*{\heiti\zihao{-2}致~~~~~~~~谢}}
	\vspace{8bp}
	\markboth{致谢}{致谢}
	\addcontentsline{toc}{chapter}{\heiti 致谢}
}{}

%重新定义附录环境
\renewenvironment{appendix}{
	{\centering\chapter*{\heiti\zihao{-2}附~~~~~~~~录}}
	\vspace{8bp}
	\markboth{附录}{附录}
	\addcontentsline{toc}{chapter}{\heiti 附录}
}{}

%定义中文摘要环境
%%页眉部分加章标题的问题尚未解决
\newenvironment{abstract}{
	\chapter*{\heiti\zihao{-2}中文摘要}
	\vspace{8bp}
	\markboth{中文摘要}{中文摘要}
	\addcontentsline{toc}{chapter}{\heiti 中文摘要}
}{}

%定义英文摘要环境
\newenvironment{englishabstract}{
	\chapter*{\zihao{-2}ABSTRACT}
	\vspace{8bp}
	\markboth{英文摘要}{英文摘要}
	\addcontentsline{toc}{chapter}{ABSTRACT}
}{}

%定义中英文关键词命令
\newcommand{\keywords}[1]{\textbf{关键词：}#1}
\newcommand{\englishkeywords}[1]{\textbf{KEYWORDS:~}#1}

%%定义正文的页眉页脚格式
\def\BJTU@chapter{}
\fancypagestyle{BJTU@heading}{
	\fancyhf{}
	\fancyhead[L]{
		\zhongsong\zihao{4}北京交通大学毕业设计（论文）
	}
	\fancyhead[R]{\zhongsong\zihao{4}\leftmark}
	\renewcommand\headrule{%
    \hrule\@height0.75pt\@width\headwidth
    \vskip 0.75pt
    \hrule\@height2.25pt\@width\headwidth
    \vskip-3.75pt
  }
	\fancyfoot[C]{\thepage}
}



%%设置页边距和装订线
\usepackage[
top=3cm,
bottom=2.5cm,
left=2.5cm,
right=2.5cm,
bindingoffset=0cm,
headheight=1.5cm,
]{geometry}

%% 设置章节格式
\ctexset{chapter={
		name={},
		number = {\arabic{chapter}},
		format = {\centering \heiti \zihao{3}},
		pagestyle = {BJTU@heading},
		beforeskip = 32bp,
		afterskip = 25bp,
		fixskip = true,
	}
}

%% 设置一级章节格式
\ctexset{section={
		format={\raggedright \bfseries \heiti \zihao{-3}},
		beforeskip = 32bp plus 1ex minus .2ex,
		afterskip = 25bp plus .2ex,
		fixskip = true,
	}
}

% 设置二级标题格式
\ctexset{subsection={
		format = {\bfseries \heiti \raggedright \zihao{4}},
		beforeskip =32bp plus 1ex minus .2ex,
		afterskip = 25bp plus .2ex,
		fixskip = true,
	}
}

% 设置三节标题格式：黑体小四居左书写，单倍行距，序号与题目间空一个汉字符
\ctexset{subsubsection={
		format={\heiti \raggedright \zihao{-4}},
		beforeskip=32bp plus 1ex minus .2ex,
		afterskip=25bp plus .2ex,
		fixskip=true,
	}
}


%目录
\ifxetex  % xelatex
\RequirePackage[xetex]{hyperref}
\hypersetup{
	bookmarksnumbered,
	colorlinks,
	linkcolor=black,
	citecolor=black,
	plainpages=false,
	pdfstartview=FitH
}
\else
\ifpdf
\RequirePackage[pdftex,unicode]{hyperref}
\else
\RequirePackage[dvipdfmx,unicode]{hyperref}
\fi
\fi

%\addtocontents{toc}{\protect\hypersetup{hidelinks}}
\addtocontents{lot}{\protect\hypersetup{hidelinks}}
\addtocontents{lof}{\protect\hypersetup{hidelinks}}

\RequirePackage{titletoc}
\titlecontents{chapter}[0pt]{\heiti \zihao{-4} \vspace{5pt}}
{\thecontentslabel\hspace{\ccwd}}{}
{\hspace{.5em}\titlerule*{.}\contentspage}
\titlecontents{section}[2\ccwd]{\songti \zihao{-4} \vspace{5pt}}
{\thecontentslabel\hspace{\ccwd}}{}
{\hspace{.5em}\titlerule*{.}\contentspage}
\titlecontents{subsection}[4\ccwd]{\songti \zihao{-4} \vspace{5pt}}
{\thecontentslabel\hspace{\ccwd}}{}
{\hspace{.5em}\titlerule*{.}\contentspage}

\titlecontents{figure}[0pt]{\songti\zihao{-4}}
{\figurename~\thecontentslabel\quad}{\hspace*{-1.5cm}}
{\hspace{.5em}\titlerule*{.}\contentspage}

\titlecontents{table}[0pt]{\songti\zihao{-4}}
{\tablename~\thecontentslabel\quad}{\hspace*{-1.5cm}}
{\hspace{.5em}\titlerule*{.}\contentspage}

\renewcommand\tableofcontents{%
	\chapter*{\heiti\zihao{-2}\contentsname}%目录里显示“目录”，否则\chapter*
	\@mkboth{\MakeUppercase\contentsname}{\MakeUppercase\contentsname}%
	\pdfbookmark[0]{目录}{bittoc}
	\@starttoc{toc}%
}



%设定文档页眉页脚
\pagestyle{BJTU@heading}
%必须在设置完文档的页眉页脚格式之后再重定义此命令
\renewcommand{\chaptermark}[1]{
	\markboth{#1}{}
}


%%定义制作论文封面命令
\newcommand{\makecover}{
	\newpage\thispagestyle{empty}
	{\color{white}~}
	\vspace{100pt}
	\begin{figure}
		\centering
		\includegraphics*[]{vi/logo.png}
	\end{figure}
	{\color{white}~}
	\vspace{-105pt}
	\begin{center}
		{\songti\zihao{2}\bfseries{本科毕业设计(论文)\\}}
		\vspace{70pt}
		{\heiti\zihao{-2}\bfseries{\BJTU@value@title\\}}
		\vspace{35pt}
		{\heiti\zihao{-2}\bfseries{\BJTU@value@englishtitle\\}}
		\vspace{55pt}
		\renewcommand\arraystretch{0.7}
		\begin{tabular}{l}
			{\songti\zihao{3}学~~~~~~~~院：\underline{~~~~~~~~\BJTU@value@school~~~~~~~~}}\\
			\\
			{\songti\zihao{3}专~~~~~~~~业：\underline{~~~~~~~~\BJTU@value@major~~~~~~~~}}\\
			\\
			{\songti\zihao{3}学生姓名：\underline{~~~~~~~~~~~~\BJTU@value@author~~~~~~~~~~~~}}\\
			\\
			{\songti\zihao{3}学~~~~~~~~号：\underline{~~~~~~~~\BJTU@value@studentNumber~~~~~~~~}}\\
			\\
			{\songti\zihao{3}指导教师：\underline{~~~~~~~~~~\BJTU@value@advisor~~~~~~~~~~}}\\
		\end{tabular}
		\\
%		\vspace{120pt}
		\vspace{55pt}
		{\songti\zihao{-3}{\bfseries{北京交通大学}}\\\vspace{14pt}\zihao{4}2021年6月}
	\end{center}
}


%% graphics packages
\RequirePackage{graphicx}
%% 并列子图
\RequirePackage{subfigure}

\RequirePackage{wrapfig}
%%===========================设置图表标题选项==========================
\RequirePackage{amsmath}
\RequirePackage{caption}
\DeclareCaptionLabelSeparator{zhspace}{\hspace{1\ccwd}}
\DeclareCaptionFont{fontsize}{\zihao{5}}
\captionsetup{
	font = {fontsize},
	labelsep = zhspace,
}
\captionsetup[table]{
	position = top,
	aboveskip = 6bp,
	belowskip = 6bp,
}
\numberwithin{table}{chapter}
\captionsetup[figure]{
	position = bottom,
	aboveskip = 6bp,
	belowskip = 6bp,
}

%% 如果插入的图片没有指定扩展名，那么依次搜索下面的扩展名所对应的文件
\DeclareGraphicsExtensions{.pdf,.eps,.png,.jpg,.jpeg}

%% sort and compress citations
\RequirePackage[numbers,square,comma,super,sort&compress]{natbib}
% 上标引用
\newcommand{\upcite}[1]{\textsuperscript{\cite{#1}}}


% 将浮动参数设为较宽松的值
\renewcommand{\textfraction}{0.15}
\renewcommand{\topfraction}{0.85}
\renewcommand{\bottomfraction}{0.65}
\renewcommand{\floatpagefraction}{0.60}


% 定公式、图、表编号为"3-1"的形式，即分隔符由.变为短杠
\renewcommand\theequation{\arabic{chapter}-\arabic{equation}}
\renewcommand\thefigure{\arabic{chapter}-\arabic{figure}}
\renewcommand\thetable{\arabic{chapter}-\arabic{table}}

\renewcommand{\ref}[1]{\zihao{-4}\ref{#1}}

% 颜色宏包
\RequirePackage{xcolor}


% 中文破折号
\newcommand{\cndash}{\rule{0.0em}{0pt}\rule[0.35em]{1.4em}{0.05em}\rule{0.2em}{0pt}}

% listings 源代码显示宏包
\RequirePackage{listings}

% 定义代码样式
\definecolor{gray}{rgb}{0.96,0.96,0.96}

\lstset{ %
  language=python,                % the language of the code
  basicstyle=\footnotesize,           % the size of the fonts that are used for the code
  numbers=left,                   % where to put the line-numbers
  columns=fixed, 
  numberstyle=\tiny\color{black},  % the style that is used for the line-numbers
  stepnumber=1,                   % the step between two line-numbers. If it's 1, each line 
                                  % will be numbered
  numbersep=5pt,                  % how far the line-numbers are from the code
  backgroundcolor=\color{gray},      % choose the background color. You must add \usepackage{color}
  showspaces=false,               % show spaces adding particular underscores
  showstringspaces=false,         % underline spaces within strings
  showtabs=false,                 % show tabs within strings adding particular underscores
  frame=single,,                 % adds a frame around the code
  frameround = tttt,
  framexleftmargin=3mm, 
  rulecolor=\color[RGB]{158,193,243},        % if not set, the frame-color may be changed on line-breaks within not-black text (e.g. commens (green here))
%  aboveskip=1em,
  tabsize=2,                      % sets default tabsize to 2 spaces
  captionpos=b,                   % sets the caption-position to bottom
  breaklines=true,                % sets automatic line breaking
  extendedchars=false,   
  breakatwhitespace=false,        % sets if automatic breaks should only happen at whitespace
  title=\lstname,                   % show the filename of files included with \lstinputlisting;
                                  % also try caption instead of title
  keywordstyle=\color[RGB]{0,51,179},          % keyword style
  commentstyle=\color[RGB]{140,140,140},       % comment style
  stringstyle=\color[RGB]{6,125,23},         % string literal style
  identifierstyle=\color{black},
  escapeinside={\%*}{*)},            % if you want to add LaTeX within your code
  morekeywords={*,...}               % if you want to add more keywords to the set
}

\renewcommand{\lstlistingname}{代码} %% 重命名Listings标题头

%% enumerate 列表环境间距调节
\RequirePackage{enumitem}
\setenumerate[1]{itemsep=0pt,partopsep=0pt,parsep=\parskip,topsep=0pt}
\setitemize[1]{itemsep=0pt,partopsep=0pt,parsep=\parskip,topsep=0pt}
\setdescription{itemsep=0pt,partopsep=0pt,parsep=\parskip,topsep=0pt}
 
 %% 伪代码
%\RequirePackage{textcomp} %命令\textacutedbl的包,二阶导符号
%\RequirePackage[linesnumbered,ruled,vlined]{algorithm2e}%[ruled,vlined]{
%\RequirePackage{algpseudocode}
%\renewcommand{\algorithmicrequire}{\textbf{Input:}}
%\renewcommand{\algorithmicensure}{\textbf{Output:}}


%_  参考文献风格 added by wei.jianwen@gmail.com
\bibliographystyle{GBT7714-2005NLang}

%_ 参考文献环境
\renewenvironment{thebibliography}[1]
{\zihao{5}
	{\centering\chapter*{\heiti \bibname}}
	\@mkboth{\MakeUppercase\bibname}{\MakeUppercase\bibname}%
	\addcontentsline{toc}{chapter}{\heiti 参考文献}
	\list{\@biblabel{\@arabic\c@enumiv}}%
	{\settowidth\labelwidth{\@biblabel{#1}}%
		\leftmargin\labelwidth
		\advance\leftmargin\labelsep
		\setlength{\parsep}{1mm}
		\setlength{\labelsep}{0.5em}
		\setlength{\itemsep}{0.05pc}
		\setlength{\listparindent}{0in}
		\setlength{\itemindent}{0in}
		\setlength{\rightmargin}{0in}
		\@openbib@code
		\usecounter{enumiv}%
		\let\p@enumiv\@empty
		\renewcommand\theenumiv{\@arabic\c@enumiv}}%
	\sloppy
	\clubpenalty4000
	\@clubpenalty \clubpenalty
	\widowpenalty4000%
	\sfcode`\.\@m}
{\def\@noitemerr
	{\@latex@warning{Empty `thebibliography' environment}}%
	\endlist}

\endinput