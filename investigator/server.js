const express = require("express");
const app = express();
const fs = require("fs");
const path = require("path");

app.use(express.static(__dirname + "/static"));

app.get("/api/data", (req, res, next) => {
  fs.readdir(__dirname + "/data", (err, files) => {
    res.send(
      files
        .filter(file => file.endsWith(".json"))
        .map(file => file.substr(0, file.length - 5))
    );
  });
});

app.get("/api/data/:id", (req, res, next) => {
  const images = JSON.parse(
    fs.readFileSync(__dirname + `/data/${req.params.id}.json`)
  );
  res.send(
    images.map(img => ({
      ...img,
      file: img.file.replace(path.join(__dirname, ".."), "")
    }))
  );
});

app.get("/api/img", (req, res) => {
  res.sendFile(path.join(__dirname, "..", req.query.file));
});

app.listen(3000);
