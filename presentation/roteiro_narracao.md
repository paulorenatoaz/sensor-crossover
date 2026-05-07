# Roteiro de Narração — Apresentação

**Duração estimada:** ~20 minutos  
**Idioma:** Português (pt-BR)

---

## Slide 1 — Título

Bom dia a todos. O tema da apresentação de hoje é "De Modelos Sintéticos a Dados Reais — Validação Empírica com Datasets Públicos".

A motivação é simples: na pesquisa em engenharia e estatística, muitos resultados teóricos são demonstrados usando dados sintéticos — dados gerados artificialmente no computador, onde controlamos tudo. A pergunta que fica é: esses resultados valem no mundo real?

Na prática, a validação com dados reais é frequentemente ausente nas publicações. O objetivo desta apresentação é justamente mostrar como podemos construir essa ponte — partir de um modelo teórico, entender suas premissas, e depois testar se as conclusões se mantêm quando usamos dados públicos reais.

Vamos ver dois projetos concretos que seguem essa filosofia.

---

## Slide 2 — Motivação (Artigo rejeitado)

Para contextualizar, vou contar uma história real. Submetemos um artigo a uma revista de referência na área, e ele foi rejeitado. As críticas dos revisores foram bastante claras e, na verdade, válidas.

Primeiro, disseram que o modelo era idealizado demais — usávamos dados Gaussianos, perfeitamente controlados. Segundo, argumentaram que o fenômeno que investigamos — o trade-off entre número de features e tamanho da amostra — já era conhecido na teoria. Terceiro, e mais importante: faltavam dados reais. Por fim, criticaram a falta de conexão com uma aplicação concreta, como redes de sensores.

Isso nos levou a uma pergunta que guia todo o restante desta apresentação: os resultados teóricos que obtivemos aparecem de fato no mundo real? Em dados ruidosos, não-Gaussianos, com todas as imperfeições de uma coleta real?

---

## Slide 3 — Abordagem

A abordagem que adotamos segue quatro passos. Primeiro, partimos do modelo sintético — os dados controlados, onde sabemos exatamente como foram gerados. Segundo, buscamos entender profundamente as premissas: o que o modelo assume e por quê.

Terceiro, construímos experimentos com dados reais, públicos e reproduzíveis. E quarto, comparamos: o que a teoria previu acontece na prática?

Uma distinção importante: no mundo sintético, temos controle total — conhecemos a distribuição, os parâmetros, tudo é reprodutível. No mundo real, temos ruído, outliers, dados faltantes, e a distribuição dos dados é desconhecida — possivelmente não-Gaussiana. Essa diferença é fundamental e é o que torna a validação relevante.

---

## Slide 4 — Projeto 1: Teoria (Trade-off entre Features e Amostras)

Vamos ao primeiro projeto. A teoria parte de um cenário de classificação binária — queremos separar dados em duas classes usando um classificador linear, especificamente um SVM linear.

No modelo sintético, geramos vetores aleatórios seguindo uma distribuição normal bivariada. Cada classe tem uma média diferente, e controlamos dois parâmetros: delta, que mede a discriminatividade — ou seja, o quanto as duas classes estão separadas — e rho, que mede a correlação entre as features.

O resultado principal é elegante: existe um ponto de cruzamento n-estrela. Quando temos poucos dados de treinamento — um n menor que n-estrela — usar menos features é melhor. É contra-intuitivo, mas faz sentido: com poucos dados, estimar uma fronteira de decisão em dimensão maior é mais difícil, e o erro aumenta. Quando temos dados suficientes — n maior que n-estrela — aí sim, mais features ajudam.

O valor de n-estrela depende de delta e rho. Essa é a previsão teórica. Agora, será que ela se sustenta com dados reais?

---

## Slide 5a — Projeto 1: Validação (Intel Lab Dataset)

Para a validação, usamos o dataset do Intel Berkeley Research Lab. São 54 sensores de temperatura distribuídos em um laboratório, com aproximadamente 2,3 milhões de leituras ao longo de 38 dias.

A construção do experimento funciona assim: escolhemos um sensor de referência, que chamamos de R — no nosso caso, o mote 22. Calculamos a mediana da temperatura desse sensor. Leituras abaixo da mediana viram classe 0, e acima viram classe 1. Essa é a nossa tarefa de classificação.

Os outros sensores são tratados como features. Selecionamos automaticamente um sensor primário A — o mais correlacionado com R — e três sensores secundários B com níveis diferentes de correlação: alto, médio e baixo. Na tabela, vemos os motes selecionados e suas correlações com o sensor de referência.

Cada combinação de sensor B define um cenário diferente, simulando as diferentes condições de correlação que o modelo teórico previa.

---

## Slide 5b — Projeto 1: Resultados Empíricos

Os resultados são muito interessantes. Rodamos 500 repetições de Monte Carlo para cada tamanho de amostra n, variando de 2 a 1024. Em cada repetição, treinamos um SVM linear usando 1 sensor ou 2 sensores e medimos o erro no conjunto de teste.

No cenário de alta correlação, o crossover acontece em n-estrela igual a 106. Com menos de 106 amostras, usar apenas o sensor A é melhor. Acima de 106, usar A e B juntos é melhor.

No cenário de baixa correlação, o crossover é mais rápido: n-estrela igual a 14. Isso faz sentido — quando o segundo sensor traz informação complementar, menos dados bastam para que ele ajude.

O mais surpreendente é o cenário de correlação média: observamos um crossover duplo, em n igual a 4 e n igual a 78. Isso é raro e foi uma das descobertas mais interessantes do trabalho.

A conclusão é clara: o fenômeno teórico — o crossover entre 1 e 2 features — persiste em dados reais. Mesmo com dados não-Gaussianos, ruidosos e com todas as imperfeições de uma rede de sensores real. Isso responde diretamente às críticas dos revisores.

---

## Slide 6 — Projeto 2: Teoria (Detecção de Mudança Multi-Sensor)

Agora vamos ao segundo projeto, que trata de um problema diferente mas com uma filosofia semelhante: detecção de mudança multi-sensor.

O modelo é o seguinte: temos um vetor de observações que segue uma normal multivariada com média mu-zero. Em algum instante desconhecido tau, a média muda para mu-um. O objetivo é detectar essa mudança o mais rápido possível.

A complicação prática é que temos múltiplos sensores correlacionados, e observar todos tem um custo — custo de sensoriamento e custo de comunicação. Então precisamos selecionar quais sensores observar.

A teoria prevê três regimes de orçamento: com orçamento baixo, observamos apenas 1 sensor; com orçamento médio, um subconjunto; e com orçamento alto, todos. O ponto crucial é que a correlação entre sensores define um trade-off entre custo e desempenho.

---

## Slide 6b — Projeto 2: Trade-offs na Detecção

Aprofundando um pouco, o problema de otimização consiste em minimizar o atraso de detecção — ou seja, o tempo entre a mudança real e a sua detecção — sujeito a uma restrição de taxa de falso alarme, e com um orçamento limitado de sensores.

A correlação entre sensores tem um papel central. Quando os sensores são muito correlacionados, a informação é redundante, e selecionar poucos sensores pode ser quase tão bom quanto usar todos. Quando são pouco correlacionados, cada sensor traz informação nova, e cortar sensores tem um custo maior.

Há uma conexão interessante com o Projeto 1: ambos tratam de seleção de sensores, e ambos mostram que mais sensores não é necessariamente melhor. No Projeto 1, a restrição é o número de amostras; no Projeto 2, é o orçamento. O insight comum é que informação extra só ajuda se tivermos recursos suficientes para explorá-la.

---

## Slide 7 — Projeto 2: Validação (SensorScope)

Para validar as previsões teóricas do Projeto 2, propomos usar o dataset SensorScope, da EPFL. São dados ambientais reais coletados por uma rede de sensores sem fio — temperatura, umidade, velocidade do vento, entre outras variáveis.

Esse dataset é ideal porque atende todos os requisitos do modelo: temos múltiplos sensores, correlação natural entre variáveis ambientais, temporalidade — os dados são séries temporais — e, claro, ruído real.

O plano experimental é: identificar mudanças reais nos dados — por exemplo, mudanças de regime de temperatura — e então variar o número de sensores ativos. Vamos comparar o atraso de detecção com diferentes subconjuntos de sensores e verificar se as predições teóricas sobre a relação entre orçamento, correlação e desempenho se confirmam na prática.

---

## Slide 8 — Extensão: CSI / Widar 3.0 (Cenário adversarial)

Uma limitação dos datasets discutidos até agora é que as mudanças nos sinais são relativamente visíveis. No Intel Lab, a temperatura varia claramente; no SensorScope, mudanças climáticas são perceptíveis. Mas e se a mudança for quase invisível? Esse é o cenário adversarial.

Para isso, propomos uma extensão usando CSI — Channel State Information. O CSI mede o efeito do ambiente físico no sinal Wi-Fi. Quando alguém se move numa sala, o sinal Wi-Fi se altera de maneira sutil. Os dados são indiretos, de alta dimensionalidade e com muito ruído.

Usamos o dataset Widar 3.0, da Universidade de Tsinghua. Em termos do modelo teórico, o cenário CSI corresponde a uma situação onde mu-um se aproxima de mu-zero — a mudança existe, mas é quase imperceptível.

A ideia é criar uma progressão: o SensorScope representa o cenário base, onde as mudanças são claras. O CSI representa o cenário adversarial, onde as mudanças são sutis. Se os resultados teóricos se mantiverem mesmo nesse cenário difícil, isso fortalece muito a validade do modelo.

---

## Slide 9 — Conclusão

Para concluir, o insight principal desta apresentação é que resultados teóricos podem, sim, persistir fora do modelo ideal. Mesmo quando relaxamos as premissas — quando os dados não são Gaussianos, quando há ruído, quando a correlação é empírica em vez de controlada — os fenômenos preditos pela teoria aparecem.

Nossas contribuições incluem: a validação empírica do fenômeno de crossover usando o Intel Lab; a proposta de validação da detecção de mudança multi-sensor com o SensorScope; e a extensão para um cenário adversarial com CSI.

Os dois projetos são complementares e seguem a mesma filosofia: partir da teoria, entender suas premissas, e testar com dados reais.

A mensagem final é simples: validação com dados reais fortalece a relevância prática dos modelos teóricos. Se a teoria funciona apenas no laboratório sintético, ela é interessante mas limitada. Se ela sobrevive ao contato com dados reais, ela é robusta e útil.

Obrigado pela atenção. Estou à disposição para perguntas.
