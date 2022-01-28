library(ggplot2)
library(dplyr)
library(reshape)
library(cowplot)
library(RColorBrewer)
library(viridis)
install.packages('ggh4x')
library(ggh4x)

models1 <- c('vgae', 'genx', 'gen')
models2 <- c('VGAE', 'GEN_crossent', 'GEN')
models3 <- c('Variational Graph Auto Encoders', 'GEN-XE', 'GEN-Hinge')

#datasets1 <- c('ec', 'deg', 'gol')
#datasets2 <- c('edit_cycles', 'degree_rules', 'game_of_life')
#datasets3 <- c('Edit Cycles DGP', 'Degree Rules DGP', 'Game of Life DGP')

datasets1 <- c('deg', 'gol')
datasets2 <- c('degree_rules', 'game_of_life')
datasets3 <- c('Degree rules DGP', 'Game of Life DGP')

style <- '_unique_test100_'
title <- 'Alt. risk estimation with 100 test series'

for (model in c(1,2,3)){
  for (dataset in c(1)){
    results <- read.csv2(paste0(datasets2[dataset],style,models2[model],'_results.csv'),sep='\t')
    results <- lapply(results, as.numeric)
    his <- data.frame(read.csv2(paste0('refs/',datasets1[dataset],'_',models1[model],'.csv')))
    his <- lapply(his, as.numeric)
    for (name in names(results)){
      results[[name]] <- results[[name]] - his[[name]]
    }
    names(results) <- c("ins_rcl", "ins_prec", "del_rcl", "del_prec",
                        "eins_rcl", "eins_prec", "edel_rcl", "edel_prec")
    

    write.csv(results, file=paste0("deltas/deltas_",datasets1[dataset],style,models1[model],".csv"))
  }
  
}

for (model in c(1,2,3)){
  for (dataset in c(1,2)){
    
    df <- read.csv2(paste0("deltas/deltas_",datasets1[dataset],style,models1[model],".csv"), sep=',')
    df <- lapply(df, as.numeric)
    df <- as.data.frame(df)
    #df <- df %>% select(-c('X'))
    
    df <- df %>% select(-c('X'))
    mdata <- melt(df, id=c())
    pl <-ggplot(data=mdata) + geom_boxplot(aes(x=variable,y=value, fill=variable)) +
      #ggtitle(paste0('Deltas: ',datasets3[dataset], '\n', models3[model],'\n', title)) +
      ylim(-0.1, 0.05) + geom_hline(yintercept=0, linetype="dashed", color = "red") + 
      theme_bw() + xlab('metric') + ylab(expression(paste(delta,': (reproduction - reported)'))) + theme(legend.position = "none") +
      scale_x_discrete(labels=c(expression(paste(bold('ins: '), 'rcl   ')), 'prc', expression(paste(bold('del: '), 'rcl   ')), 'prc', expression(paste(bold('eins: '), 'rcl   ')), 'prc', expression(paste(bold('edel: '), 'rcl   ')), 'prc'))
    
    
    filename <- paste0('./deltas/plots/',datasets1[dataset],style,models1[model],'.png')
    ggsave(filename,plot=pl, height=1080, width=2000, units='px', bg='white')
    
    
  }
}

get_method <- function(label) {
  c <- as.character(label)
  noquote(strsplit(c, '.', fixed=TRUE)[[1]][1])
}

get_pass <- function(label) {
  c <- as.character(label)
  strsplit(c, '.', fixed=TRUE)[[1]][2]
}

human_numbers <- function(x = NULL, smbl ="", signif = 1){
  humanity <- function(y){
    
    if (!is.na(y)){
      tn <- round(abs(y) / 1e12, signif)
      b <- round(abs(y) / 1e9, signif)
      m <- round(abs(y) / 1e6, signif)
      k <- round(abs(y) / 1e3, signif)
      
      if ( y >= 0 ){
        y_is_positive <- ""
      } else {
        y_is_positive <- "-"
      }
      
      if ( k < 1 ) {
        paste0( y_is_positive, smbl, round(abs(y), signif ))
      } else if ( m < 1){
        paste0 (y_is_positive, smbl,  k , "k")
      } else if (b < 1){
        paste0 (y_is_positive, smbl, m ,"m")
      }else if(tn < 1){
        paste0 (y_is_positive, smbl, b ,"bn")
      } else {
        paste0 (y_is_positive, smbl,  comma(tn), "tn")
      }
    } else if (is.na(y) | is.null(y)){
      "-"
    }
  }
  
  sapply(x,humanity)
}
human_num   <- function(x){human_numbers(x, smbl = "")} 



df <- melt(read.csv('hep_th_runtimes_conv_1.csv', sep=","), id='Graph.size')
ggplot(data = df, aes(x=Graph.size, y=value)) + 
  geom_point(alpha=0.1, aes(color=variable)) + 
  facet_wrap(~variable, nrow=2)+
  geom_smooth(method='loess', color='black') + scale_y_continuous(trans='log10') +
  scale_x_continuous(trans='log10', labels = human_num) + 
  ylab('Runtime [s]') + xlab('graph node count') +
  theme_bw() + theme(legend.position="none") + theme(strip.text.x = element_text(
    face = "bold"
  ))

scales_x <- list(
  'boolean' = scale_x_discrete(xmin=1, xmax=6),
  'peano' = scale_x_discrete(xmin=1, xmax=70)
)
scales = list(x = scales_x)
ddf <- read.csv('penosBooleans.csv')
ggplot(data = ddf) + geom_histogram(aes(x=value, fill=kind)) + facet_grid(variable~kind, scales = "free", space='free_x' ) +
  force_panelsizes(cols = c(0.3, 1))+
  theme_bw() + ggtitle('Distribution of sampled TS lengths') + xlab('Number of graphs in TS') + ylab('Count of sampled TS')