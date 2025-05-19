FROM golang:1.20

WORKDIR /app
COPY . .

RUN go mod tidy
RUN go build -o main .

# Add this line to copy the script
COPY wait_for_test_csv.sh /wait_for_test_csv.sh

# Make it executable (optional if you're doing chmod locally)
RUN chmod +x /wait_for_test_csv.sh

# Update entrypoint
CMD ["/wait_for_test_csv.sh"]