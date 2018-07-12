let SessionLoad = 1
if &cp | set nocp | endif
let s:cpo_save=&cpo
set cpo&vim
imap <S-Tab> <Plug>snipMateBack
inoremap <C-Tab> 	
inoremap <silent> <Plug>snipMateShow =snipMate#ShowAvailableSnips()
inoremap <silent> <Plug>snipMateBack =snipMate#BackwardsSnippet()
inoremap <silent> <Plug>snipMateTrigger =snipMate#TriggerSnippet(1)
inoremap <silent> <Plug>snipMateNextOrTrigger =snipMate#TriggerSnippet()
map  :CtrlPBuffer
map  <Plug>(ctrlp)
map  h
xmap 	 <Plug>snipMateVisual
smap 	 <Plug>snipMateNextOrTrigger
snoremap <NL> i<Right>=snipMate#TriggerSnippet()
nmap <NL> j
xmap <NL> j
omap <NL> j
map  k
map  l
nmap  <Plug>yankstack_substitute_newer_paste
nmap  <Plug>yankstack_substitute_older_paste
xnoremap <silent>  :call multiple_cursors#new("v", 0)
nnoremap <silent>  :call multiple_cursors#new("n", 1)
nnoremap  :call GotoFile("new")
nnoremap f :call GotoFile("new")
map   /
vnoremap <silent> # :call VisualSelection('', '')?=@/
vnoremap $e `>a"`<i"
vnoremap $q `>a'`<i'
vnoremap $$ `>a"`<i"
vnoremap $3 `>a}`<i{
vnoremap $2 `>a]`<i[
vnoremap $1 `>a)`<i(
vnoremap <silent> * :call VisualSelection('', '')/=@/
vmap + <Plug>(expand_region_expand)
nmap + <Plug>(expand_region_expand)
nmap ,hp <Plug>GitGutterPreviewHunk
nmap ,hu <Plug>GitGutterUndoHunk
nmap ,hs <Plug>GitGutterStageHunk
map ,p :cp
map ,n :cn
map ,co ggVGy:tabnew:set syntax=qfpgg
map ,cc :botright cope
vnoremap <silent> ,r :call VisualSelection('replace', '')
map ,g :Ack 
map ,e :e! ~/.vim_runtime/my_configs.vim
nnoremap <silent> ,d :GitGutterToggle
nmap <silent> ,a <Plug>(ale_next_wrap)
nnoremap <silent> ,z :Goyo
map ,nf :NERDTreeFind
map ,nb :NERDTreeFromBookmark 
map ,nn :NERDTreeToggle
map ,j :CtrlP
map ,f :MRU
map ,o :BufExplorer
map ,pp :setlocal paste!
map ,x :e ~/buffer.md
map ,q :e ~/buffer
noremap ,m mmHmt:%s///ge'tzt'm
map ,s? z=
map ,sa zg
map ,sp [s
map ,sn ]s
map ,ss :setlocal spell!
map ,cd :cd %:p:h:pwd
map ,te :tabedit =expand("%:p:h")/
nmap ,tl :exe "tabn ".g:lasttab
map ,t, :tabnext 
map ,tm :tabmove 
map ,tc :tabclose
map ,to :tabonly
map ,tn :tabnew
map ,h :bprevious
map ,l :bnext
map ,ba :bufdo bd
map ,bd :Bclose:tabclosegT
map <silent> , :noh
nmap ,w :w!
nmap 0 ^
xmap 0 ^
omap 0 ^
imap ½ $
cmap ½ $
imap Ð <Plug>yankstack_substitute_newer_paste
imap ð <Plug>yankstack_substitute_older_paste
xmap S <Plug>VSurround
vmap Si S(i_f)
nmap [c <Plug>GitGutterPrevHunk
nmap ]c <Plug>GitGutterNextHunk
nmap _ <Plug>(expand_region_shrink)
xmap _ <Plug>(expand_region_shrink)
xmap ac <Plug>GitGutterTextObjectOuterVisual
omap ac <Plug>GitGutterTextObjectOuterPending
nmap cS <Plug>CSurround
nmap cs <Plug>Csurround
nmap cgc <Plug>ChangeCommentary
nmap cr <Plug>Coerce
nmap ds <Plug>Dsurround
vmap gx <Plug>NetrwBrowseXVis
nmap gx <Plug>NetrwBrowseX
xmap gS <Plug>VgSurround
nmap gcu <Plug>Commentary<Plug>Commentary
nmap gcc <Plug>CommentaryLine
omap gc <Plug>Commentary
nmap gc <Plug>Commentary
xmap gc <Plug>Commentary
nnoremap gf :call GotoFile("")
vnoremap <silent> gv :call VisualSelection('gv', '')
xmap ic <Plug>GitGutterTextObjectInnerVisual
omap ic <Plug>GitGutterTextObjectInnerPending
nmap ySS <Plug>YSsurround
nmap ySs <Plug>YSsurround
nmap yss <Plug>Yssurround
nmap yS <Plug>YSurround
nmap ys <Plug>Ysurround
smap <S-Tab> <Plug>snipMateBack
vnoremap <silent> <Plug>NetrwBrowseXVis :call netrw#BrowseXVis()
nnoremap <silent> <Plug>NetrwBrowseX :call netrw#BrowseX(expand((exists("g:netrw_gx")? g:netrw_gx : '<cfile>')),netrw#CheckIfRemote())
nnoremap <silent> <Plug>SurroundRepeat .
snoremap <silent> <Plug>snipMateBack a=snipMate#BackwardsSnippet()
snoremap <silent> <Plug>snipMateNextOrTrigger a=snipMate#TriggerSnippet()
nnoremap <silent> <Plug>GitGutterPreviewHunk :GitGutterPreviewHunk
nnoremap <silent> <Plug>GitGutterUndoHunk :GitGutterUndoHunk
nnoremap <silent> <Plug>GitGutterStageHunk :GitGutterStageHunk
nnoremap <silent> <expr> <Plug>GitGutterPrevHunk &diff ? '[c' : ":\execute v:count1 . 'GitGutterPrevHunk'\"
nnoremap <silent> <expr> <Plug>GitGutterNextHunk &diff ? ']c' : ":\execute v:count1 . 'GitGutterNextHunk'\"
xnoremap <silent> <Plug>GitGutterTextObjectOuterVisual :call gitgutter#hunk#text_object(0)
xnoremap <silent> <Plug>GitGutterTextObjectInnerVisual :call gitgutter#hunk#text_object(1)
onoremap <silent> <Plug>GitGutterTextObjectOuterPending :call gitgutter#hunk#text_object(0)
onoremap <silent> <Plug>GitGutterTextObjectInnerPending :call gitgutter#hunk#text_object(1)
nnoremap <SNR>70_: :=v:count ? v:count : ''
xnoremap <silent> <Plug>(expand_region_shrink) :call expand_region#next('v', '-')
xnoremap <silent> <Plug>(expand_region_expand) :call expand_region#next('v', '+')
nnoremap <silent> <Plug>(expand_region_expand) :call expand_region#next('n', '+')
nmap <silent> <Plug>CommentaryUndo <Plug>Commentary<Plug>Commentary
nnoremap <silent> <Plug>(ctrlp) :CtrlP
nnoremap <silent> <Plug>(ale_go_to_definition_in_tab) :ALEGoToDefinitionInTab
nnoremap <silent> <Plug>(ale_go_to_definition) :ALEGoToDefinition
nnoremap <silent> <Plug>(ale_fix) :ALEFix
nnoremap <silent> <Plug>(ale_detail) :ALEDetail
nnoremap <silent> <Plug>(ale_lint) :ALELint
nnoremap <silent> <Plug>(ale_reset_buffer) :ALEResetBuffer
nnoremap <silent> <Plug>(ale_disable_buffer) :ALEDisableBuffer
nnoremap <silent> <Plug>(ale_enable_buffer) :ALEEnableBuffer
nnoremap <silent> <Plug>(ale_toggle_buffer) :ALEToggleBuffer
nnoremap <silent> <Plug>(ale_reset) :ALEReset
nnoremap <silent> <Plug>(ale_disable) :ALEDisable
nnoremap <silent> <Plug>(ale_enable) :ALEEnable
nnoremap <silent> <Plug>(ale_toggle) :ALEToggle
nnoremap <silent> <Plug>(ale_last) :ALELast
nnoremap <silent> <Plug>(ale_first) :ALEFirst
nnoremap <silent> <Plug>(ale_next_wrap) :ALENextWrap
nnoremap <silent> <Plug>(ale_next) :ALENext
nnoremap <silent> <Plug>(ale_previous_wrap) :ALEPreviousWrap
nnoremap <silent> <Plug>(ale_previous) :ALEPrevious
map <C-Space> ?
cnoremap  <Home>
cnoremap  <End>
imap S <Plug>ISurround
imap s <Plug>Isurround
imap 	 <Plug>snipMateNextOrTrigger
inoremap <NL> =snipMate#TriggerSnippet()
cnoremap  
cnoremap  <Down>
cnoremap  <Up>
imap 	 <Plug>snipMateShow
imap  <Plug>Isurround
inoremap $e ""i
inoremap $q ''i
inoremap $4 {o}O
inoremap $3 {}i
inoremap $2 []i
inoremap $1 ()i
cnoremap $q eDeleteTillSlash()
cnoremap $c e eCurrentFileDir("e")
cnoremap $j e ./
cnoremap $d e ~/Desktop/
cnoremap $h e ~/
map! <silent> ,p <Plug>PeepOpen
map ½ $
xmap Ð <Plug>yankstack_substitute_newer_paste
nmap Ð <Plug>yankstack_substitute_newer_paste
xmap ð <Plug>yankstack_substitute_older_paste
nmap ð <Plug>yankstack_substitute_older_paste
vmap ë :m'<-2`>my`<mzgv`yo`z
vmap ê :m'>+`<my`>mzgv`yo`z
nmap ë mz:m-2`z
nmap ê mz:m+`z
iabbr xdate =strftime("%d/%m/%y %H:%M:%S")
let &cpo=s:cpo_save
unlet s:cpo_save
set autoindent
set autoread
set background=dark
set backspace=eol,start,indent
set cindent
set cinkeys=0{,0},0),:,!^F,o,O,e
set cmdheight=2
set expandtab
set fileencodings=ucs-bom,utf-8,default,latin1
set fileformats=unix,dos,mac
set grepprg=/bin/grep\ -nH
set helplang=en
set hidden
set history=500
set hlsearch
set ignorecase
set incsearch
set indentkeys=0{,0},:,!^F,o,O,e,<:>,=elif,=except
set isfname=@,48-57,/,.,-,_,+,,,#,$,%,~,=,:
set langmenu=en
set laststatus=2
set lazyredraw
set matchtime=2
set nomodeline
set printoptions=paper:a4
set ruler
set runtimepath=~/.vim_runtime/sources_non_forked/ack.vim,~/.vim_runtime/sources_non_forked/ale,~/.vim_runtime/sources_non_forked/bufexplorer,~/.vim_runtime/sources_non_forked/ctrlp.vim,~/.vim_runtime/sources_non_forked/goyo.vim,~/.vim_runtime/sources_non_forked/gruvbox,~/.vim_runtime/sources_non_forked/lightline.vim,~/.vim_runtime/sources_non_forked/mayansmoke,~/.vim_runtime/sources_non_forked/mru.vim,~/.vim_runtime/sources_non_forked/nerdtree,~/.vim_runtime/sources_non_forked/nginx.vim,~/.vim_runtime/sources_non_forked/open_file_under_cursor.vim,~/.vim_runtime/sources_non_forked/rust.vim,~/.vim_runtime/sources_non_forked/snipmate-snippets,~/.vim_runtime/sources_non_forked/tabular,~/.vim_runtime/sources_non_forked/tlib,~/.vim_runtime/sources_non_forked/vim-abolish,~/.vim_runtime/sources_non_forked/vim-addon-mw-utils,~/.vim_runtime/sources_non_forked/vim-bundle-mako,~/.vim_runtime/sources_non_forked/vim-coffee-script,~/.vim_runtime/sources_non_forked/vim-colors-solarized,~/.vim_runtime/sources_non_forked/vim-commentary,~/.vim_runtime/sources_non_forked/vim-expand-region,~/.vim_runtime/sources_non_forked/vim-flake8,~/.vim_runtime/sources_non_forked/vim-fugitive,~/.vim_runtime/sources_non_forked/vim-gitgutter,~/.vim_runtime/sources_non_forked/vim-go,~/.vim_runtime/sources_non_forked/vim-indent-object,~/.vim_runtime/sources_non_forked/vim-less,~/.vim_runtime/sources_non_forked/vim-markdown,~/.vim_runtime/sources_non_forked/vim-multiple-cursors,~/.vim_runtime/sources_non_forked/vim-pug,~/.vim_runtime/sources_non_forked/vim-pyte,~/.vim_runtime/sources_non_forked/vim-repeat,~/.vim_runtime/sources_non_forked/vim-snipmate,~/.vim_runtime/sources_non_forked/vim-snippets,~/.vim_runtime/sources_non_forked/vim-surround,~/.vim_runtime/sources_non_forked/vim-yankstack,~/.vim_runtime/sources_non_forked/vim-zenroom2,~/.vim_runtime/sources_forked/peaksea,~/.vim_runtime/sources_forked/set_tabline,~/.vim_runtime/sources_forked/vim-irblack-forked,~/.vim_runtime/sources_forked/vim-peepopen,~/.vim,/var/lib/vim/addons,/usr/share/vim/vimfiles,/usr/share/vim/vim80,/usr/share/vim/vimfiles/after,/var/lib/vim/addons/after,~/.vim/after,~/.vim_runtime,~/.vim_runtime/sources_non_forked/vim-snipmate/after,~/.vim_runtime/sources_non_forked/vim-markdown/after,~/.vim_runtime/sources_non_forked/vim-less/after,~/.vim_runtime/sources_non_forked/vim-coffee-script/after,~/.vim_runtime/sources_non_forked/tabular/after,~/.vim_runtime/sources_non_forked/rust.vim/after,~/.vim_runtime/sources_non_forked/ale/after
set scrolloff=7
set shiftwidth=4
set showmatch
set showtabline=2
set smartcase
set smartindent
set smarttab
set statusline=\ %{HasPaste()}%F%m%r%h\ %w\ \ CWD:\ %r%{getcwd()}%h\ \ \ Line:\ %l\ \ Column:\ %c
set suffixes=.bak,~,.swp,.o,.info,.aux,.log,.dvi,.bbl,.blg,.brf,.cb,.ind,.idx,.ilg,.inx,.out,.toc
set noswapfile
set switchbuf=useopen,usetab,newtab
set tabline=%!lightline#tabline()
set tabstop=4
set textwidth=500
set timeoutlen=500
set undodir=~/.vim_runtime/temp_dirs/undodir
set undofile
set whichwrap=b,s,<,>,h,l
set wildignore=*.o,*~,*.pyc,*/.git/*,*/.hg/*,*/.svn/*,*/.DS_Store
set wildmenu
set nowritebackup
let s:so_save = &so | let s:siso_save = &siso | set so=0 siso=0
let v:this_session=expand("<sfile>:p")
silent only
cd ~/Desktop/new/research/capsule/SegCaps-Pytorch
if expand('%') == '' && !&modified && line('$') <= 1 && getline(1) == ''
  let s:wipebuf = bufnr('%')
endif
set shortmess=aoO
badd +0 models/capsule_layers.py
badd +1 ~/Desktop/new/research/capsule/SegCaps/capsule_layers.py
argglobal
silent! argdel *
$argadd models/capsule_layers.py
edit models/capsule_layers.py
set splitbelow splitright
set nosplitbelow
set nosplitright
wincmd t
set winminheight=1 winheight=1 winminwidth=1 winwidth=1
argglobal
map <buffer> ,D ?def 
map <buffer> ,C ?class 
map <buffer> ,2 /def 
map <buffer> ,1 /class 
map <buffer> F :set foldmethod=indent
let s:cpo_save=&cpo
set cpo&vim
noremap <buffer> <F7> :call flake8#Flake8()
inoremap <buffer> $f # --- a
inoremap <buffer> $p print 
inoremap <buffer> $i import 
inoremap <buffer> $r return 
let &cpo=s:cpo_save
unlet s:cpo_save
setlocal keymap=
setlocal noarabic
setlocal autoindent
setlocal backupcopy=
setlocal balloonexpr=
setlocal nobinary
setlocal nobreakindent
setlocal breakindentopt=
setlocal bufhidden=
setlocal buflisted
setlocal buftype=
setlocal cindent
setlocal cinkeys=0{,0},0),:,!^F,o,O,e
setlocal cinoptions=
setlocal cinwords=if,else,while,do,for,switch
setlocal colorcolumn=
setlocal comments=b:#,fb:-
setlocal commentstring=#\ %s
setlocal complete=.,w,b,u,t,i
setlocal concealcursor=
setlocal conceallevel=0
setlocal completefunc=
setlocal nocopyindent
setlocal cryptmethod=
setlocal nocursorbind
setlocal nocursorcolumn
setlocal nocursorline
setlocal define=
setlocal dictionary=
setlocal nodiff
setlocal equalprg=
setlocal errorformat=
setlocal expandtab
if &filetype != 'python'
setlocal filetype=python
endif
setlocal fixendofline
set foldcolumn=1
setlocal foldcolumn=1
setlocal foldenable
setlocal foldexpr=0
setlocal foldignore=#
setlocal foldlevel=0
setlocal foldmarker={{{,}}}
setlocal foldmethod=manual
setlocal foldminlines=1
setlocal foldnestmax=20
setlocal foldtext=foldtext()
setlocal formatexpr=
setlocal formatoptions=tcq
setlocal formatlistpat=^\\s*\\d\\+[\\]:.)}\\t\ ]\\s*
setlocal formatprg=
setlocal grepprg=
setlocal iminsert=0
setlocal imsearch=-1
setlocal include=^\\s*\\(from\\|import\\)
setlocal includeexpr=substitute(substitute(substitute(v:fname,b:grandparent_match,b:grandparent_sub,''),b:parent_match,b:parent_sub,''),b:child_match,b:child_sub,'g')
setlocal indentexpr=GetPythonIndent(v:lnum)
setlocal indentkeys=0{,0},:,!^F,o,O,e,<:>,=elif,=except
setlocal noinfercase
setlocal iskeyword=@,48-57,_,192-255
setlocal keywordprg=pydoc
set linebreak
setlocal linebreak
setlocal nolisp
setlocal lispwords=
setlocal nolist
setlocal makeencoding=
setlocal makeprg=
setlocal matchpairs=(:),{:},[:]
setlocal nomodeline
setlocal modifiable
setlocal nrformats=bin,octal,hex
setlocal nonumber
setlocal numberwidth=4
setlocal omnifunc=python3complete#Complete
setlocal path=
setlocal nopreserveindent
setlocal nopreviewwindow
setlocal quoteescape=\\
setlocal noreadonly
setlocal norelativenumber
setlocal norightleft
setlocal rightleftcmd=search
setlocal noscrollbind
setlocal shiftwidth=4
setlocal noshortname
setlocal signcolumn=auto
setlocal smartindent
setlocal softtabstop=4
setlocal nospell
setlocal spellcapcheck=[.?!]\\_[\\])'\"\	\ ]\\+
setlocal spellfile=
setlocal spelllang=en
setlocal statusline=%{lightline#link()}%#LightlineLeft_active_0#%(\ %{lightline#mode()}\ %)%{(&paste)?\"\ \":\"\"}%(\ %{&paste?\"PASTE\":\"\"}\ %)%#LightlineLeft_active_0_1#\ %#LightlineLeft_active_1#%(\ %{exists(\"*fugitive#head\")?fugitive#head():\"\"}\ %)%{((exists(\"*fugitive#head\")\ &&\ \"\"!=fugitive#head()))&&(((&filetype!=\"help\"&&\ &readonly))||1||((&filetype!=\"help\"&&(&modified||!&modifiable))))?\"\ \":\"\"}%(\ %{&filetype==\"help\"?\"\":&readonly?\"🔒\":\"\"}\ %)%{((&filetype!=\"help\"&&\ &readonly))&&(1||((&filetype!=\"help\"&&(&modified||!&modifiable))))?\"\ \":\"\"}%(\ %t\ %)%{((&filetype!=\"help\"&&(&modified||!&modifiable)))?\"\ \":\"\"}%(\ %{&filetype==\"help\"?\"\":&modified?\"+\":&modifiable?\"\":\"-\"}\ %)%#LightlineLeft_active_1_2#\ %#LightlineMiddle_active#%=%#LightlineRight_active_1_2#\ %#LightlineRight_active_1#%(\ %3p%%\ %)%#LightlineRight_active_0_1#\ %#LightlineRight_active_0#%(\ %3l:%-2v\ %)
setlocal suffixesadd=.py
setlocal noswapfile
setlocal synmaxcol=3000
if &syntax != 'python'
setlocal syntax=python
endif
setlocal tabstop=8
setlocal tagcase=
setlocal tags=
setlocal termwinkey=
setlocal termwinscroll=10000
setlocal termwinsize=
setlocal textwidth=500
setlocal thesaurus=
setlocal undofile
setlocal undolevels=-123456
setlocal nowinfixheight
setlocal nowinfixwidth
setlocal wrap
setlocal wrapmargin=0
silent! normal! zE
let s:l = 57 - ((26 * winheight(0) + 16) / 33)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
57
normal! 09|
tabedit ~/Desktop/new/research/capsule/SegCaps/capsule_layers.py
set splitbelow splitright
set nosplitbelow
set nosplitright
wincmd t
set winminheight=1 winheight=1 winminwidth=1 winwidth=1
argglobal
map <buffer> ,D ?def 
map <buffer> ,C ?class 
map <buffer> ,2 /def 
map <buffer> ,1 /class 
map <buffer> F :set foldmethod=indent
let s:cpo_save=&cpo
set cpo&vim
noremap <buffer> <F7> :call flake8#Flake8()
inoremap <buffer> $f # --- a
inoremap <buffer> $p print 
inoremap <buffer> $i import 
inoremap <buffer> $r return 
let &cpo=s:cpo_save
unlet s:cpo_save
setlocal keymap=
setlocal noarabic
setlocal autoindent
setlocal backupcopy=
setlocal balloonexpr=
setlocal nobinary
setlocal nobreakindent
setlocal breakindentopt=
setlocal bufhidden=
setlocal buflisted
setlocal buftype=
setlocal cindent
setlocal cinkeys=0{,0},0),:,!^F,o,O,e
setlocal cinoptions=
setlocal cinwords=if,else,while,do,for,switch
setlocal colorcolumn=
setlocal comments=b:#,fb:-
setlocal commentstring=#\ %s
setlocal complete=.,w,b,u,t,i
setlocal concealcursor=
setlocal conceallevel=0
setlocal completefunc=
setlocal nocopyindent
setlocal cryptmethod=
setlocal nocursorbind
setlocal nocursorcolumn
setlocal nocursorline
setlocal define=
setlocal dictionary=
setlocal nodiff
setlocal equalprg=
setlocal errorformat=
setlocal expandtab
if &filetype != 'python'
setlocal filetype=python
endif
setlocal fixendofline
set foldcolumn=1
setlocal foldcolumn=1
setlocal foldenable
setlocal foldexpr=0
setlocal foldignore=#
setlocal foldlevel=0
setlocal foldmarker={{{,}}}
setlocal foldmethod=manual
setlocal foldminlines=1
setlocal foldnestmax=20
setlocal foldtext=foldtext()
setlocal formatexpr=
setlocal formatoptions=tcq
setlocal formatlistpat=^\\s*\\d\\+[\\]:.)}\\t\ ]\\s*
setlocal formatprg=
setlocal grepprg=
setlocal iminsert=0
setlocal imsearch=-1
setlocal include=^\\s*\\(from\\|import\\)
setlocal includeexpr=substitute(substitute(substitute(v:fname,b:grandparent_match,b:grandparent_sub,''),b:parent_match,b:parent_sub,''),b:child_match,b:child_sub,'g')
setlocal indentexpr=GetPythonIndent(v:lnum)
setlocal indentkeys=0{,0},:,!^F,o,O,e,<:>,=elif,=except
setlocal noinfercase
setlocal iskeyword=@,48-57,_,192-255
setlocal keywordprg=pydoc
set linebreak
setlocal linebreak
setlocal nolisp
setlocal lispwords=
setlocal nolist
setlocal makeencoding=
setlocal makeprg=
setlocal matchpairs=(:),{:},[:]
setlocal nomodeline
setlocal modifiable
setlocal nrformats=bin,octal,hex
setlocal nonumber
setlocal numberwidth=4
setlocal omnifunc=python3complete#Complete
setlocal path=
setlocal nopreserveindent
setlocal nopreviewwindow
setlocal quoteescape=\\
setlocal noreadonly
setlocal norelativenumber
setlocal norightleft
setlocal rightleftcmd=search
setlocal noscrollbind
setlocal shiftwidth=4
setlocal noshortname
setlocal signcolumn=auto
setlocal smartindent
setlocal softtabstop=4
setlocal nospell
setlocal spellcapcheck=[.?!]\\_[\\])'\"\	\ ]\\+
setlocal spellfile=
setlocal spelllang=en
setlocal statusline=%{lightline#link()}%#LightlineLeft_active_0#%(\ %{lightline#mode()}\ %)%{(&paste)?\"\ \":\"\"}%(\ %{&paste?\"PASTE\":\"\"}\ %)%#LightlineLeft_active_0_1#\ %#LightlineLeft_active_1#%(\ %{exists(\"*fugitive#head\")?fugitive#head():\"\"}\ %)%{((exists(\"*fugitive#head\")\ &&\ \"\"!=fugitive#head()))&&(((&filetype!=\"help\"&&\ &readonly))||1||((&filetype!=\"help\"&&(&modified||!&modifiable))))?\"\ \":\"\"}%(\ %{&filetype==\"help\"?\"\":&readonly?\"🔒\":\"\"}\ %)%{((&filetype!=\"help\"&&\ &readonly))&&(1||((&filetype!=\"help\"&&(&modified||!&modifiable))))?\"\ \":\"\"}%(\ %t\ %)%{((&filetype!=\"help\"&&(&modified||!&modifiable)))?\"\ \":\"\"}%(\ %{&filetype==\"help\"?\"\":&modified?\"+\":&modifiable?\"\":\"-\"}\ %)%#LightlineLeft_active_1_2#\ %#LightlineMiddle_active#%=%#LightlineRight_active_1_2#\ %#LightlineRight_active_1#%(\ %3p%%\ %)%#LightlineRight_active_0_1#\ %#LightlineRight_active_0#%(\ %3l:%-2v\ %)
setlocal suffixesadd=.py
setlocal noswapfile
setlocal synmaxcol=3000
if &syntax != 'python'
setlocal syntax=python
endif
setlocal tabstop=8
setlocal tagcase=
setlocal tags=
setlocal termwinkey=
setlocal termwinscroll=10000
setlocal termwinsize=
setlocal textwidth=500
setlocal thesaurus=
setlocal undofile
setlocal undolevels=-123456
setlocal nowinfixheight
setlocal nowinfixwidth
setlocal wrap
setlocal wrapmargin=0
silent! normal! zE
let s:l = 46 - ((25 * winheight(0) + 16) / 33)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
46
normal! 0
tabnext 1
if exists('s:wipebuf') && s:wipebuf != bufnr('%')
  silent exe 'bwipe ' . s:wipebuf
endif
unlet! s:wipebuf
set winheight=1 winwidth=20 shortmess=filnxtToO
set winminheight=1 winminwidth=1
let s:sx = expand("<sfile>:p:r")."x.vim"
if file_readable(s:sx)
  exe "source " . fnameescape(s:sx)
endif
let &so = s:so_save | let &siso = s:siso_save
doautoall SessionLoadPost
unlet SessionLoad
" vim: set ft=vim :
