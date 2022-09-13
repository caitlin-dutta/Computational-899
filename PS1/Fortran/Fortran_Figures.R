valuefun$pf_h <- polfun$X2
valuefun$pf_l <- polfun$X3

colnames(valuefun) <- c("Capital", "high_VF", "low_VF", "high_PF", "low_PF")
ggplot(valuefun) +   geom_line(aes(x=Capital,y=high_VF, group=1, colour = 'High productivity')) +    geom_line(aes(x=Capital,y=low_VF, group = 2, colour = 'Low productivity')) +   ylab('Value')+xlab('Capital') + ggtitle("Value functions by productivity type")
ggplot(valuefun) +   geom_line(aes(x=Capital,y=high_PF, group=1, colour = 'High productivity')) +    geom_line(aes(x=Capital,y=low_PF, group = 2, colour = 'Low productivity')) +   ylab('Policy')+xlab('Capital') + ggtitle("Policy functions by productivity type")
