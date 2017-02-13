require 'set'


directory 'var'


task :dataset => 'var' do |t|
  repo_dir = File.join t.source, 'clojure'
  sh "git clone https://github.com/clojure/clojure #{repo_dir}" unless File.directory? repo_dir

  dataset_dir = File.join(t.source, 'dataset')
  mkdir_p dataset_dir

  filenames = Dir.glob('var/clojure/**/*.clj')
  p filenames.length

  [
    ['train', filenames[0..-22]],
    ['dev', filenames[-21..-12]],
    ['test', filenames[-11..-1]],
  ].each do |subdir, fnames|
    dirname = File.join dataset_dir, subdir
    mkdir_p dirname
    fnames.each do |fname|
      cp fname , File.join(dirname, fname.gsub(/\//, '_'))
    end
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
