## Rust の知識

### 継承（Derive）

- コンパイラには、`[#derive]` アトリビュートを用いることで方に対しての特定のトレイトの標準的な実装を提供する機能がある。
- 構造体などに振る舞いを追加することができる
- derive可能なトレイトの一覧
  - Eq: 同じ値での比較がすべての値で真になる場合に付与できるトレイト（浮動小数点型ではNaN同士の比較が真にならないので実装されていない）
  - PartialEq: オブジェクト同士が透過であるかの比較を行うためのトレイト
  - Clone: コピーによって `&T` から `T` を作成するトレイト
  - Hash: `&T` からハッシュ値を計算するためのトレイト
  - Debug: `{:?}` フォーマッタを利用して値をフォーマットするためのトレイト
- [Rustの構造体などに追加できる振る舞いを確認する - Qiita](https://qiita.com/apollo_program/items/2495dda519ae160971ed)

### impl

- Implement some functionally for a type.
- The `impl` keyword is primarily used to define implementations on types. Inherent implementations are standalone, while trait implementations are used to implement traits for types, ot other traits.
- [impl - Rust](https://doc.rust-lang.org/beta/std/keyword.impl.html)

```rust
struct Example {
  number: i32,
}

impl Example {
  fn boo() {
    println!("oo!)
  }
}
```

## Parser の知識

### 字句解析

- 字句解析とは、ユーザーの入力した文字列からトークン列を切り出す操作。
- 後に続く構文解析の前段の処理で、トークン列にすることで空白の虫や文字列から数値への変換などの文字列処理を終わらせる。

### トークン

- 言語の文法のうち、終端記号と呼ばれるものがトークン
- トークンとは、文法において直接文字列として表現されている要素

