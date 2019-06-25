:syntax on
:set ruler
:set incsearch
:set wrap linebreak nolist
:set autochdir
:set foldmethod=indent
:set foldlevel=99
:set clipboard=unnamed
:set ts=2 sts=2 sw=2 expandtab
:set cc=89
autocmd Filetype python setlocal ts=4 sts=4 sw=4
autocmd BufWritePre * :%s/\s\+$//e
