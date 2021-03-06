# kwiieiya - ボイスロイド向け歌唱転写ツール

kwiieiya は VOICEROID 向け歌唱転写ツールです。
VOCALOID 等の歌唱を VOICEROID の音声に歌唱転写することで VOICEROID に歌わせることを目標としています。

### とりあえず使うには？
1. 歌わせたいボイロ等の音声を 入力 の 開く ボタンを押して開きます
2. CARRIER のチェックをオンにし、転写元となるボカロ等で生成した歌の音声を 開く ボタンを押して開きます
3. 一番下の 再生 ボタンを押して結果を確認します
4. 保存 ボタンを押して結果を保存します

### こんな時には？
* 音声ファイルを読み込むとエラーになってしまう
  * 対応している音声形式か確認してください
    * フォーマット : wav
    * チャンネル数 : 1 ch
      * 音声がモノラルでない場合、最初のチャンネルの音声が使用されます
    * ビット深度 : 8bit 整数 , 16 bit 整数 , 32 bit 浮動小数点
      * 24 bit 整数形式には非対応
* 入力 と CARRIER の音声のタイミングの同期がとれていない
  * サンプリングレートが 入力 と CARRIER で異なっていないか確認してください
  * 入力 や CARRIER の 再生 ボタンを押し、正しく分析できているか確認してください
    * ノイズが混ざっている場合は正しく分析できていません
    * どのような音声で分析が失敗したか報告していただけると、改善に繋がるかもしれません
  * 入力の音声に伸ばし棒を入れるなどしてキャリアの音声のタイミングに近づけてみてください
  * 音声ファイルを分割し、短くしてみてください
