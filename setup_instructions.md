# Setup instructions to run this blog locally

1. Install `bundler`.
2. Install the required libraries for `nokogiri`:

```bash
sudo apt-get install build-essential patch ruby-dev zlib1g-dev liblzma-dev
```

3. Run `bundle install` and `bundle update` inside the blog directory.
4. Execute with `bundle exec jekyll serve` to get a localhost instance of the blog.
