const sortings = {
  none: (a, b) => -1,
  cer: (a, b) => b.cer - a.cer,
  class: (a, b) => a.classification - b.classification
};

class App extends React.Component {
  constructor() {
    super();
    this.state = {
      files: [],
      selectedFile: "",
      images: [],
      sorting: "none"
    };
    fetch("/api/data")
      .then(res => res.json())
      .then(data => {
        this.setState({
          files: data
        });
        if (data.length > 0) {
          this.selectFile(data[0]);
        }
      });
  }

  selectFile(selectedFile) {
    this.setState({
      selectedFile
    });
    fetch(`/api/data/${selectedFile}`)
      .then(res => res.json())
      .then(data => {
        this.setState({
          images: data
        });
      });
  }

  onSelectChange(event) {
    this.selectFile(event.target.value);
  }
  onSortingChange(event) {
    this.setState({ sorting: event.target.value });
  }

  render() {
    return (
      <div class="container">
        <div class="controller">
          <select
            onChange={target => this.onSelectChange(target)}
            value={this.state.selectedFile}
          >
            {this.state.files.map(file => (
              <option value={file} key={file}>
                {file}
              </option>
            ))}
          </select>

          <select
            onChange={target => this.onSortingChange(target)}
            value={this.state.sorting}
          >
            <option value="none">None</option>
            <option value="cer">CER</option>
            <option value="class">Classification</option>
          </select>
        </div>
        <div class="images">
          <table>
            <thead>
              <tr>
                <th>Image</th>
                <th>Transcription</th>
                <th>C. Error</th>
                <th>Handwriting?</th>
              </tr>
            </thead>
            <tbody>
              {this.state.images
                .sort(sortings[this.state.sorting])
                .map(image => <Image {...image} />)}
            </tbody>
          </table>
        </div>
      </div>
    );
  }
}

class Image extends React.Component {
  constructor() {
    super();
    this.state = {
      width: "?",
      height: "?"
    };
  }
  imgLoaded(image) {
    this.setState({
      width: image.naturalWidth,
      height: image.naturalHeight
    });
  }
  getColor(min, max, value, invert) {
    let r = Math.abs((value - min) / (max - min)) * 255;
    let g = (1 - Math.abs((value - min) / (max - min))) * 255;
    if (invert) {
      r = 255 - r;
      g = 255 - g;
    }
    return `rgb(${r}, ${g}, 0)`;
  }

  render() {
    const cerStyle = {
      backgroundColor: this.getColor(0, 0.4, this.props.cer)
    };
    const classStyle = {
      backgroundColor: this.getColor(0.3, 1, this.props.classification, true)
    };
    return (
      <tr>
        <td class="image">
          <img
            src={`/api/img?file=${this.props.file}`}
            onLoad={event => this.imgLoaded(event.target)}
          />
        </td>
        <td class="trans">
          <b>{this.props.transcription}</b>
          <br />
          {this.props.truth}
        </td>
        <td class="cer" style={cerStyle}>
          {Math.round(this.props.cer * 10000) / 100} %
        </td>
        <td class="class" style={classStyle}>
          {Math.round(this.props.classification * 10000) / 100} %
        </td>
        <td>
          {this.state.width} x {this.state.height}
        </td>
      </tr>
    );
  }
}

ReactDOM.render(<App />, document.getElementById("app"));
