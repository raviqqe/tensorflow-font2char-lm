require 'set'


directory 'var'


task :dataset => 'var' do |t|
  repo_dir = File.join t.source, 'clojure'
  sh "git clone https://github.com/clojure/clojure #{repo_dir}" unless File.directory? repo_dir

  dataset_dir = File.join(t.source, 'dataset')
  mkdir_p dataset_dir
  Dir.glob('**/*.clj').each do |filename|
    cp filename , File.join(dataset_dir, filename.gsub(/\//, '_'))
  end
end


file 'var/chars_tmp.txt' => :dataset do |t|
  chars = ''

  Dir.glob('var/dataset/*.clj').each do |filename|
    chars += File.read(filename).gsub(/[[:space:]]/, '').gsub(/[^[:print:]]/, '')
  end

  File.write t.name, Set.new(chars.chars).to_a.sort.join.gsub(/./, "\\0\n")
end


file 'var/chars.txt' => 'var/chars_tmp.txt' do |t|
  sh "echo '<none>' > #{t.name}"
  sh "echo '<unknown>' >> #{t.name}"
  sh "echo '<s>' >> #{t.name}"
  sh "echo '</s>' >> #{t.name}"
  sh "cat #{t.source} >> #{t.name}"
end
